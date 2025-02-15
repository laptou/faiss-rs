//! Interface and implementation to RefineFlat index type.

use super::*;

use std::marker::PhantomData;
use std::os::raw::c_int;

/// Alias for the native implementation of a index.
pub type RefineFlatIndex<BI> = RefineFlatIndexImpl<BI>;

/// Native implementation of a RefineFlat index.
#[derive(Debug)]
pub struct RefineFlatIndexImpl<BI> {
    inner: *mut FaissIndexRefineFlat,
    base_index: PhantomData<BI>,
}

unsafe impl<BI: Send> Send for RefineFlatIndexImpl<BI> {}
unsafe impl<BI: Sync> Sync for RefineFlatIndexImpl<BI> {}

impl<BI: CpuIndex> CpuIndex for RefineFlatIndexImpl<BI> {}

impl<BI> Drop for RefineFlatIndexImpl<BI> {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexRefineFlat_free(self.inner);
        }
    }
}

impl<BI: NativeIndex> RefineFlatIndexImpl<BI> {
    pub fn new(base_index: BI) -> Result<Self> {
        let index = RefineFlatIndexImpl::new_helper(&base_index, true)?;
        mem::forget(base_index);
        Ok(index)
    }

    fn new_helper<I: NativeIndex>(base_index: &I, own_fields: bool) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexRefineFlat_new(
                &mut inner,
                base_index.inner_ptr(),
            ))?;
            faiss_IndexRefineFlat_set_own_fields(inner, c_int::from(own_fields));
            Ok(RefineFlatIndexImpl {
                inner,
                base_index: PhantomData,
            })
        }
    }

    pub fn set_k_factor(&mut self, kf: f32) {
        unsafe {
            faiss_IndexRefineFlat_set_k_factor(self.inner_ptr(), kf);
        }
    }

    pub fn k_factor(&self) -> f32 {
        unsafe { faiss_IndexRefineFlat_k_factor(self.inner_ptr()) }
    }
}

impl<BI> NativeIndex for RefineFlatIndexImpl<BI> {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for RefineFlatIndexImpl<IndexImpl> {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        RefineFlatIndexImpl {
            inner: inner_ptr as *mut FaissIndexFlat,
            base_index: PhantomData,
        }
    }
}

impl TryFromInnerPtr for RefineFlatIndexImpl<IndexImpl> {
    unsafe fn try_from_inner_ptr(inner_ptr: *mut FaissIndex) -> Result<Self>
    where
        Self: Sized,
    {
        // safety: `inner_ptr` is documented to be a valid pointer to an index,
        // so the dynamic cast should be safe.
        #[allow(unused_unsafe)]
        unsafe {
            let new_inner = faiss_IndexRefineFlat_cast(inner_ptr);
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                Ok(RefineFlatIndexImpl {
                    inner: new_inner,
                    base_index: PhantomData,
                })
            }
        }
    }
}

impl_native_index!(RefineFlatIndexImpl<I>, I);
impl_concurrent_index!(RefineFlatIndexImpl<I>, I: ConcurrentIndex);

impl<I> TryClone for RefineFlatIndexImpl<I> {
    fn try_clone(&self) -> Result<Self>
    where
        Self: Sized,
    {
        unsafe {
            let mut new_index_ptr = ::std::ptr::null_mut();
            faiss_try(faiss_clone_index(self.inner_ptr(), &mut new_index_ptr))?;
            Ok(RefineFlatIndexImpl {
                inner: new_index_ptr as *mut FaissIndexFlat,
                base_index: PhantomData,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RefineFlatIndexImpl;
    use crate::index::{flat::FlatIndexImpl, ConcurrentIndex, Idx, Index, UpcastIndex};

    const D: u32 = 8;

    #[test]
    fn refine_flat_index_search() {
        let index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let mut refine = RefineFlatIndexImpl::new(index).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        refine.add(some_data).unwrap();
        assert_eq!(refine.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = refine.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&refine).search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        refine.reset().unwrap();
        assert_eq!(refine.ntotal(), 0);
    }

    #[test]
    fn refine_flat_index_upcast() {
        let index = FlatIndexImpl::new_l2(D).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let refine = RefineFlatIndexImpl::new(index).unwrap();

        let index_impl = refine.upcast();
        assert_eq!(index_impl.d(), D);
    }
}

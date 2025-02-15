//! Implementation to PreTransform index type.

use super::*;

use crate::vector_transform::NativeVectorTransform;
use std::marker::PhantomData;
use std::os::raw::c_int;

/// Alias for the native implementation of a PreTransform index.
pub type PreTransformIndex<I> = PreTransformIndexImpl<I>;

/// Native implementation of a flat index.
#[derive(Debug)]
pub struct PreTransformIndexImpl<I> {
    inner: *mut FaissIndexPreTransform,
    sub_index: PhantomData<I>,
}

unsafe impl<I: Send> Send for PreTransformIndexImpl<I> {}
unsafe impl<I> Sync for PreTransformIndexImpl<I> {}

impl<I: CpuIndex> CpuIndex for PreTransformIndexImpl<I> {}

impl<I> Drop for PreTransformIndexImpl<I> {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexPreTransform_free(self.inner);
        }
    }
}

impl<I> PreTransformIndexImpl<I>
where
    I: NativeIndex,
{
    pub fn new<LT: NativeVectorTransform>(lt: LT, sub_index: I) -> Result<Self> {
        let index = PreTransformIndexImpl::new_helper(&lt, &sub_index, true)?;
        mem::forget(lt);
        mem::forget(sub_index);

        Ok(index)
    }

    fn new_helper<LT: NativeVectorTransform>(
        lt: &LT,
        sub_index: &I,
        own_fields: bool,
    ) -> Result<Self> {
        unsafe {
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexPreTransform_new_with_transform(
                &mut inner,
                lt.inner_ptr(),
                sub_index.inner_ptr(),
            ))?;
            faiss_IndexPreTransform_set_own_fields(inner, c_int::from(own_fields));
            Ok(PreTransformIndexImpl {
                inner,
                sub_index: PhantomData,
            })
        }
    }

    pub fn prepend_transform<LT: NativeVectorTransform>(&mut self, ltrans: LT) -> Result<()> {
        unsafe {
            faiss_try(faiss_IndexPreTransform_prepend_transform(
                self.inner,
                ltrans.inner_ptr(),
            ))?;

            Ok(())
        }
    }
}

impl IndexImpl {
    pub fn into_pre_transform(self) -> Result<PreTransformIndexImpl<IndexImpl>> {
        unsafe {
            let new_inner = faiss_IndexPreTransform_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(PreTransformIndexImpl {
                    inner: new_inner,
                    sub_index: PhantomData,
                })
            }
        }
    }
}

impl<I> NativeIndex for PreTransformIndexImpl<I> {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for PreTransformIndexImpl<IndexImpl> {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        PreTransformIndexImpl {
            inner: inner_ptr as *mut FaissIndexPreTransform,
            sub_index: PhantomData,
        }
    }
}

impl_index!(PreTransformIndexImpl<I>, I);
impl_concurrent_index!(PreTransformIndexImpl<I>, I: ConcurrentIndex);

impl<I> TryClone for PreTransformIndexImpl<I> {
    fn try_clone(&self) -> Result<Self>
    where
        Self: Sized,
    {
        unsafe {
            let mut new_index_ptr = ::std::ptr::null_mut();
            faiss_try(faiss_clone_index(self.inner_ptr(), &mut new_index_ptr))?;
            Ok(PreTransformIndexImpl {
                inner: new_index_ptr as *mut FaissIndexPreTransform,
                sub_index: PhantomData,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::index::pretransform::PreTransformIndexImpl;
    use crate::index::UpcastIndex as _;
    use crate::metric::MetricType;
    use crate::{
        index::{index_factory, ConcurrentIndex, Idx, Index},
        vector_transform::PCAMatrixImpl,
    };

    const D: u32 = 8;

    #[test]
    fn pre_transform_index_from_cast_upcast() {
        let mut index = index_factory(D, "PCA4,Flat", MetricType::L2).unwrap();

        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        if !index.is_trained() {
            index.train(some_data).unwrap();
        }
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index = index.into_pre_transform().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
        assert_eq!(index.d(), 8);

        let index_impl = index.upcast();
        assert_eq!(index_impl.is_trained(), true);
        assert_eq!(index_impl.ntotal(), 5);
        assert_eq!(index_impl.d(), 8);
    }

    #[test]
    fn pre_transform_index_search() {
        const D_OUT: u32 = D / 2;
        let index = crate::index::flat::FlatIndexImpl::new_l2(D_OUT).unwrap();
        assert_eq!(index.d(), D_OUT);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];

        let vt = PCAMatrixImpl::new(D, D_OUT, 0f32, false).unwrap();
        let mut pre_transform_index = PreTransformIndexImpl::new(vt, index).unwrap();
        assert_eq!(pre_transform_index.d(), D);

        if !pre_transform_index.is_trained() {
            pre_transform_index.train(some_data).unwrap();
        }
        pre_transform_index.add(some_data).unwrap();
        assert_eq!(pre_transform_index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = pre_transform_index.search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![2, 1, 0, 3, 4]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        let result = (&pre_transform_index).search(&my_query, 5).unwrap();
        assert_eq!(
            result.labels,
            vec![3, 4, 0, 1, 2]
                .into_iter()
                .map(Idx::new)
                .collect::<Vec<_>>()
        );
        assert!(result.distances.iter().all(|x| *x > 0.));

        pre_transform_index.reset().unwrap();
        assert_eq!(pre_transform_index.ntotal(), 0);
    }
}

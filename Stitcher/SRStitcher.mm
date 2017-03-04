//
//  SRStitcher.cpp
//  Stitcher
//
//  Created by a on 2016/2/19.
//  Copyright © 2017年 a. All rights reserved.
//

#include "SRStitcher.hpp"
#import "convertTool.h"
#import "FileUtils.h"

namespace sr {
    
    SRStitcher SRStitcher::createDefault(bool try_use_gpu)
    {
        SRStitcher stitcher;
        stitcher.setRegistrationResol(0.1);
        stitcher.setSeamEstimationResol(0.1);
        stitcher.setCompositingResol(ORIG_RESOL);
        stitcher.setPanoConfidenceThresh(1);
        stitcher.setWaveCorrection(true);
        stitcher.setWaveCorrectKind(cv::detail::WAVE_CORRECT_HORIZ);
        stitcher.setFeaturesMatcher(cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_use_gpu, 0.5));
        stitcher.setBundleAdjuster(cv::makePtr<cv::detail::BundleAdjusterRay>());
        
#ifdef HAVE_CUDA
        if (try_use_gpu && cuda::getCudaEnabledDeviceCount() > 0)
        {
#ifdef HAVE_OPENCV_XFEATURES2D
            stitcher.setFeaturesFinder(makePtr<detail::SurfFeaturesFinderGpu>());
#else
            stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
            stitcher.setWarper(makePtr<SphericalWarperGpu>());
            stitcher.setSeamFinder(makePtr<detail::GraphCutSeamFinderGpu>());
        }
        else
#endif
        {
#ifdef HAVE_OPENCV_XFEATURES2D
            stitcher.setFeaturesFinder(cv::makePtr<cv::detail::SurfFeaturesFinder>());
#else
            stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
            stitcher.setWarper(cv::makePtr<cv::SphericalWarper>());
            stitcher.setSeamFinder(cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR));
        }
        
        stitcher.setExposureCompensator(cv::makePtr<cv::detail::BlocksGainCompensator>());
        stitcher.setBlender(cv::makePtr<cv::detail::MultiBandBlender>(try_use_gpu));
        
        return stitcher;
    }
    
    
    SRStitcher::Status SRStitcher::estimateTransform(cv::InputArrayOfArrays images)
    {
        return estimateTransform(images, std::vector<std::vector<cv::Rect> >());
    }
    
    
    SRStitcher::Status SRStitcher::estimateTransform(cv::InputArrayOfArrays images, const std::vector<std::vector<cv::Rect> > &rois)
    {
        images.getUMatVector(imgs_);
        rois_ = rois;
        
        Status status;
        
        if ((status = matchImages()) != OK)
            return status;
        
        if ((status = estimateCameraParams()) != OK)
            return status;
        
        return OK;
    }
    
    
    
    SRStitcher::Status SRStitcher::composePanorama(cv::OutputArray pano)
    {
        return composePanorama(std::vector<cv::UMat>(), pano);
    }
    
    
    SRStitcher::Status SRStitcher::getWrapImageAndMask(cv::InputArrayOfArrays images, size_t &warpCnt)
    {
        LOGLN("Warping images (auxiliary)... ");
        
        Status status = estimateTransform(images);
        if (status != OK)
            return status;
        
                
        cv::UMat pano_;
        
#if ENABLE_LOG
        int64 t = getTickCount();
#endif
        
        std::vector<cv::Point> corners(imgs_.size());
        std::vector<cv::UMat> masks_warped(imgs_.size());
        std::vector<cv::UMat> images_warped(imgs_.size());
        std::vector<cv::Size> sizes(imgs_.size());
        std::vector<cv::UMat> masks(imgs_.size());
        
        warpCnt = imgs_.size();
        
        // Prepare image masks
        for (size_t i = 0; i < imgs_.size(); ++i)
        {
            masks[i].create(seam_est_imgs_[i].size(), CV_8U);
            masks[i].setTo(cv::Scalar::all(255));
        }
        
        // Warp images and their masks
        cv::Ptr<cv::detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
        for (size_t i = 0; i < imgs_.size(); ++i)
        {
            cv::Mat_<float> K;
            cameras_[i].K().convertTo(K, CV_32F);
            K(0,0) *= (float)seam_work_aspect_;
            K(0,2) *= (float)seam_work_aspect_;
            K(1,1) *= (float)seam_work_aspect_;
            K(1,2) *= (float)seam_work_aspect_;
            
            corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, images_warped[i]);
            sizes[i] = images_warped[i].size();
            
            w->warp(masks[i], K, cameras_[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
        }
        
        std::vector<cv::UMat> images_warped_f(imgs_.size());
        for (size_t i = 0; i < imgs_.size(); ++i)
            images_warped[i].convertTo(images_warped_f[i], CV_32F);
        
        LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
        
        // Find seams
        exposure_comp_->feed(corners, images_warped, masks_warped);
        seam_finder_->find(images_warped_f, corners, masks_warped);
        
        // Release unused memory
        seam_est_imgs_.clear();
        images_warped.clear();
        images_warped_f.clear();
        masks.clear();
        
        LOGLN("Compositing...");
#if ENABLE_LOG
        t = getTickCount();
#endif
        
        cv::UMat img_warped, img_warped_s;
        cv::UMat dilated_mask, seam_mask, mask, mask_warped;
        
        //double compose_seam_aspect = 1;
        double compose_work_aspect = 1;
        
        double compose_scale = 1;
        bool is_compose_scale_set = false;
        
        cv::UMat full_img, img;
        NSMutableArray *nsconners = [NSMutableArray array];
        NSMutableArray *nssizes = [NSMutableArray array];

        for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
        {
            LOGLN("Compositing image #" << indices_[img_idx] + 1);
#if ENABLE_LOG
            int64 compositing_t = getTickCount();
#endif
            
            // Read image and resize it if necessary
            full_img = imgs_[img_idx];
            if (!is_compose_scale_set)
            {
                if (compose_resol_ > 0)
                    compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
                is_compose_scale_set = true;
                
                // Compute relative scales
                //compose_seam_aspect = compose_scale / seam_scale_;
                compose_work_aspect = compose_scale / work_scale_;
                
                // Update warped image scale
                warped_image_scale_ *= static_cast<float>(compose_work_aspect);
                w = warper_->create((float)warped_image_scale_);
                
                // Update corners and sizes
                for (size_t i = 0; i < imgs_.size(); ++i)
                {
                    // Update intrinsics
                    cameras_[i].focal *= compose_work_aspect;
                    cameras_[i].ppx *= compose_work_aspect;
                    cameras_[i].ppy *= compose_work_aspect;
                    
                    // Update corner and size
                    cv::Size sz = full_img_sizes_[i];
                    if (std::abs(compose_scale - 1) > 1e-1)
                    {
                        sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
                        sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
                    }
                    
                    cv::Mat K;
                    cameras_[i].K().convertTo(K, CV_32F);
                    cv::Rect roi = w->warpRoi(sz, K, cameras_[i].R);
                    corners[i] = roi.tl();
                    sizes[i] = roi.size();
                }
            }
            if (std::abs(compose_scale - 1) > 1e-1)
            {
#if ENABLE_LOG
                int64 resize_t = getTickCount();
#endif
                resize(full_img, img, cv::Size(), compose_scale, compose_scale);
                LOGLN("  resize time: " << ((getTickCount() - resize_t) / getTickFrequency()) << " sec");
            }
            else
                img = full_img;
            full_img.release();
            cv::Size img_size = img.size();
            
            LOGLN(" after resize time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");
            
            cv::Mat K;
            cameras_[img_idx].K().convertTo(K, CV_32F);
            
#if ENABLE_LOG
            int64 pt = getTickCount();
#endif
            // Warp the current image
            w->warp(img, K, cameras_[img_idx].R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, img_warped);
            LOGLN(" warp the current image: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
            pt = getTickCount();
#endif
            
            // Warp the current image mask
            mask.create(img_size, CV_8U);
            mask.setTo(cv::Scalar::all(255));
            w->warp(mask, K, cameras_[img_idx].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);
            LOGLN(" warp the current image mask: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
            pt = getTickCount();
#endif
            
            // Compensate exposure
            exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);
            LOGLN(" compensate exposure: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
            pt = getTickCount();
#endif
            
            img_warped.convertTo(img_warped_s, CV_16S);
            img_warped.release();
            img.release();
            mask.release();
            
            // Make sure seam mask has proper size
            dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
            resize(dilated_mask, seam_mask, mask_warped.size());
            
            bitwise_and(seam_mask, mask_warped, mask_warped);
            
            LOGLN(" other: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
            
            cv::UMat matToStorage;
            img_warped_s.convertTo(matToStorage, 0);
            img_warped_s.release();
            UIImage *img = [convertTool imageFromMat:matToStorage];
            UIImage *mask = [convertTool imageFromMat:mask_warped];
//
            matToStorage.release();
            mask_warped.release();
            seam_mask.release();


            NSData *imageData = UIImagePNGRepresentation(img);
            NSData *maskData = UIImagePNGRepresentation(mask);
            [imageData writeToFile:[FileUtils warpImagePath:img_idx] atomically:YES];
            [maskData writeToFile:[FileUtils maskImagePath:img_idx] atomically:YES];
        
            [nsconners addObject:[NSValue valueWithCGPoint:CGPointMake(corners[img_idx].x, corners[img_idx].y)]];
            [nssizes addObject:[NSValue valueWithCGSize:CGSizeMake(sizes[img_idx].width, sizes[img_idx].height)]];
        }

        [NSKeyedArchiver archiveRootObject:nsconners toFile:[FileUtils connersPath]];
        [NSKeyedArchiver archiveRootObject:nssizes toFile:[FileUtils sizesPath]];

        return OK;
    }
    
    SRStitcher::Status SRStitcher::composePanorama(cv::InputArrayOfArrays images, cv::OutputArray pano)
    {
        LOGLN("Warping images (auxiliary)... ");
        
        std::vector<cv::UMat> imgs;
        images.getUMatVector(imgs);
        if (!imgs.empty())
        {
            CV_Assert(imgs.size() == imgs_.size());
            
            cv::UMat img;
            seam_est_imgs_.resize(imgs.size());
            
            for (size_t i = 0; i < imgs.size(); ++i)
            {
                imgs_[i] = imgs[i];
                resize(imgs[i], img, cv::Size(), seam_scale_, seam_scale_);
                seam_est_imgs_[i] = img.clone();
            }
            
            std::vector<cv::UMat> seam_est_imgs_subset;
            std::vector<cv::UMat> imgs_subset;
            
            for (size_t i = 0; i < indices_.size(); ++i)
            {
                imgs_subset.push_back(imgs_[indices_[i]]);
                seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
            }
            
            seam_est_imgs_ = seam_est_imgs_subset;
            imgs_ = imgs_subset;
        }
        
        cv::UMat pano_;
        
#if ENABLE_LOG
        int64 t = getTickCount();
#endif
        
        std::vector<cv::Point> corners(imgs_.size());
        std::vector<cv::UMat> masks_warped(imgs_.size());
        std::vector<cv::UMat> images_warped(imgs_.size());
        std::vector<cv::Size> sizes(imgs_.size());
        std::vector<cv::UMat> masks(imgs_.size());
        
        // Prepare image masks
        for (size_t i = 0; i < imgs_.size(); ++i)
        {
            masks[i].create(seam_est_imgs_[i].size(), CV_8U);
            masks[i].setTo(cv::Scalar::all(255));
        }
        
        // Warp images and their masks
        cv::Ptr<cv::detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
        for (size_t i = 0; i < imgs_.size(); ++i)
        {
            cv::Mat_<float> K;
            cameras_[i].K().convertTo(K, CV_32F);
            K(0,0) *= (float)seam_work_aspect_;
            K(0,2) *= (float)seam_work_aspect_;
            K(1,1) *= (float)seam_work_aspect_;
            K(1,2) *= (float)seam_work_aspect_;
            
            corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, images_warped[i]);
            sizes[i] = images_warped[i].size();
            
            w->warp(masks[i], K, cameras_[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, masks_warped[i]);
        }
        
        std::vector<cv::UMat> images_warped_f(imgs_.size());
        for (size_t i = 0; i < imgs_.size(); ++i)
            images_warped[i].convertTo(images_warped_f[i], CV_32F);
        
        LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
        
        // Find seams
        exposure_comp_->feed(corners, images_warped, masks_warped);
        seam_finder_->find(images_warped_f, corners, masks_warped);
        
        // Release unused memory
        seam_est_imgs_.clear();
        images_warped.clear();
        images_warped_f.clear();
        masks.clear();
        
        LOGLN("Compositing...");
#if ENABLE_LOG
        t = getTickCount();
#endif
        
        cv::UMat img_warped, img_warped_s;
        cv::UMat dilated_mask, seam_mask, mask, mask_warped;
        
        //double compose_seam_aspect = 1;
        double compose_work_aspect = 1;
        bool is_blender_prepared = false;
        
        double compose_scale = 1;
        bool is_compose_scale_set = false;
        
        cv::UMat full_img, img;
        for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
        {
            LOGLN("Compositing image #" << indices_[img_idx] + 1);
#if ENABLE_LOG
            int64 compositing_t = getTickCount();
#endif
            
            // Read image and resize it if necessary
            full_img = imgs_[img_idx];
            if (!is_compose_scale_set)
            {
                if (compose_resol_ > 0)
                    compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
                is_compose_scale_set = true;
                
                // Compute relative scales
                //compose_seam_aspect = compose_scale / seam_scale_;
                compose_work_aspect = compose_scale / work_scale_;
                
                // Update warped image scale
                warped_image_scale_ *= static_cast<float>(compose_work_aspect);
                w = warper_->create((float)warped_image_scale_);
                
                // Update corners and sizes
                for (size_t i = 0; i < imgs_.size(); ++i)
                {
                    // Update intrinsics
                    cameras_[i].focal *= compose_work_aspect;
                    cameras_[i].ppx *= compose_work_aspect;
                    cameras_[i].ppy *= compose_work_aspect;
                    
                    // Update corner and size
                    cv::Size sz = full_img_sizes_[i];
                    if (std::abs(compose_scale - 1) > 1e-1)
                    {
                        sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
                        sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
                    }
                    
                    cv::Mat K;
                    cameras_[i].K().convertTo(K, CV_32F);
                    cv::Rect roi = w->warpRoi(sz, K, cameras_[i].R);
                    corners[i] = roi.tl();
                    sizes[i] = roi.size();
                }
            }
            if (std::abs(compose_scale - 1) > 1e-1)
            {
#if ENABLE_LOG
                int64 resize_t = getTickCount();
#endif
                resize(full_img, img, cv::Size(), compose_scale, compose_scale);
                LOGLN("  resize time: " << ((getTickCount() - resize_t) / getTickFrequency()) << " sec");
            }
            else
                img = full_img;
            full_img.release();
            cv::Size img_size = img.size();
            
            LOGLN(" after resize time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");
            
            cv::Mat K;
            cameras_[img_idx].K().convertTo(K, CV_32F);
            
#if ENABLE_LOG
            int64 pt = getTickCount();
#endif
            // Warp the current image
            w->warp(img, K, cameras_[img_idx].R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, img_warped);
            LOGLN(" warp the current image: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
            pt = getTickCount();
#endif
            
            // Warp the current image mask
            mask.create(img_size, CV_8U);
            mask.setTo(cv::Scalar::all(255));
            w->warp(mask, K, cameras_[img_idx].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);
            LOGLN(" warp the current image mask: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
            pt = getTickCount();
#endif
            
            // Compensate exposure
            exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);
            LOGLN(" compensate exposure: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
            pt = getTickCount();
#endif
            
            img_warped.convertTo(img_warped_s, CV_16S);
            img_warped.release();
            img.release();
            mask.release();
            
            // Make sure seam mask has proper size
            dilate(masks_warped[img_idx], dilated_mask, cv::Mat());
            resize(dilated_mask, seam_mask, mask_warped.size());
            
            bitwise_and(seam_mask, mask_warped, mask_warped);
            
            LOGLN(" other: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
            pt = getTickCount();
#endif
            
            if (!is_blender_prepared)
            {
                blender_->prepare(corners, sizes);
                is_blender_prepared = true;
            }
            
            LOGLN(" other2: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
            
            LOGLN(" feed...");
#if ENABLE_LOG
            int64 feed_t = getTickCount();
#endif
            // Blend the current image
            blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
            LOGLN(" feed time: " << ((getTickCount() - feed_t) / getTickFrequency()) << " sec");
            LOGLN("Compositing ## time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");
            
        }
        
#if ENABLE_LOG
        int64 blend_t = getTickCount();
#endif
        cv::UMat result, result_mask;
        blender_->blend(result, result_mask);
        LOGLN("blend time: " << ((getTickCount() - blend_t) / getTickFrequency()) << " sec");
        
        LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
        
        // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
        // so convert it to avoid user confusing
        result.convertTo(pano, CV_8U);
        
        return OK;
    }
    
    
    SRStitcher::Status SRStitcher::stitch(cv::InputArrayOfArrays images, cv::OutputArray pano)
    {
        Status status = estimateTransform(images);
        if (status != OK)
            return status;
        return composePanorama(pano);
    }
    
    
    SRStitcher::Status SRStitcher::stitch(cv::InputArrayOfArrays images, const std::vector<std::vector<cv::Rect> > &rois, cv::OutputArray pano)
    {
        Status status = estimateTransform(images, rois);
        if (status != OK)
            return status;
        return composePanorama(pano);
    }
    
    
    SRStitcher::Status SRStitcher::matchImages()
    {
        if ((int)imgs_.size() < 2)
        {
            LOGLN("Need more images");
            return ERR_NEED_MORE_IMGS;
        }
        
        work_scale_ = 1;
        seam_work_aspect_ = 1;
        seam_scale_ = 1;
        bool is_work_scale_set = false;
        bool is_seam_scale_set = false;
        cv::UMat full_img, img;
        features_.resize(imgs_.size());
        seam_est_imgs_.resize(imgs_.size());
        full_img_sizes_.resize(imgs_.size());
        
        LOGLN("Finding features...");
#if ENABLE_LOG
        int64 t = getTickCount();
#endif
        
        for (size_t i = 0; i < imgs_.size(); ++i)
        {
            full_img = imgs_[i];
            full_img_sizes_[i] = full_img.size();
            
            if (registr_resol_ < 0)
            {
                img = full_img;
                work_scale_ = 1;
                is_work_scale_set = true;
            }
            else
            {
                if (!is_work_scale_set)
                {
                    work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img.size().area()));
                    is_work_scale_set = true;
                }
                resize(full_img, img, cv::Size(), work_scale_, work_scale_);
            }
            if (!is_seam_scale_set)
            {
                seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img.size().area()));
                seam_work_aspect_ = seam_scale_ / work_scale_;
                is_seam_scale_set = true;
            }
            
            if (rois_.empty())
                (*features_finder_)(img, features_[i]);
            else
            {
                std::vector<cv::Rect> rois(rois_[i].size());
                for (size_t j = 0; j < rois_[i].size(); ++j)
                {
                    cv::Point tl(cvRound(rois_[i][j].x * work_scale_), cvRound(rois_[i][j].y * work_scale_));
                    cv::Point br(cvRound(rois_[i][j].br().x * work_scale_), cvRound(rois_[i][j].br().y * work_scale_));
                    rois[j] = cv::Rect(tl, br);
                }
                (*features_finder_)(img, features_[i], rois);
            }
            features_[i].img_idx = (int)i;
            LOGLN("Features in image #" << i+1 << ": " << features_[i].keypoints.size());
            
            resize(full_img, img, cv::Size(), seam_scale_, seam_scale_);
            seam_est_imgs_[i] = img.clone();
        }
        
        // Do it to save memory
        features_finder_->collectGarbage();
        full_img.release();
        img.release();
        
        LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
        
        LOG("Pairwise matching");
#if ENABLE_LOG
        t = getTickCount();
#endif
        (*features_matcher_)(features_, pairwise_matches_, matching_mask_);
        features_matcher_->collectGarbage();
        LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
        
        // Leave only images we are sure are from the same panorama
        indices_ = cv::detail::leaveBiggestComponent(features_, pairwise_matches_, (float)conf_thresh_);
        std::vector<cv::UMat> seam_est_imgs_subset;
        std::vector<cv::UMat> imgs_subset;
        std::vector<cv::Size> full_img_sizes_subset;
        for (size_t i = 0; i < indices_.size(); ++i)
        {
            imgs_subset.push_back(imgs_[indices_[i]]);
            seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
            full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
        }
        seam_est_imgs_ = seam_est_imgs_subset;
        imgs_ = imgs_subset;
        full_img_sizes_ = full_img_sizes_subset;
        
        if ((int)imgs_.size() < 2)
        {
            LOGLN("Need more images");
            return ERR_NEED_MORE_IMGS;
        }
        
        return OK;
    }
    
    
    SRStitcher::Status SRStitcher::estimateCameraParams()
    {
        cv::detail::HomographyBasedEstimator estimator;
        if (!estimator(features_, pairwise_matches_, cameras_))
            return ERR_HOMOGRAPHY_EST_FAIL;
        
        for (size_t i = 0; i < cameras_.size(); ++i)
        {
            cv::Mat R;
            cameras_[i].R.convertTo(R, CV_32F);
            cameras_[i].R = R;
            //LOGLN("Initial intrinsic parameters #" << indices_[i] + 1 << ":\n " << cameras_[i].K());
        }
        
        bundle_adjuster_->setConfThresh(conf_thresh_);
        if (!(*bundle_adjuster_)(features_, pairwise_matches_, cameras_))
            return ERR_CAMERA_PARAMS_ADJUST_FAIL;
        
        // Find median focal length and use it as final image scale
        std::vector<double> focals;
        for (size_t i = 0; i < cameras_.size(); ++i)
        {
            //LOGLN("Camera #" << indices_[i] + 1 << ":\n" << cameras_[i].K());
            focals.push_back(cameras_[i].focal);
        }
        
        std::sort(focals.begin(), focals.end());
        if (focals.size() % 2 == 1)
            warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
        else
            warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
        
        if (do_wave_correct_)
        {
            std::vector<cv::Mat> rmats;
            for (size_t i = 0; i < cameras_.size(); ++i)
                rmats.push_back(cameras_[i].R.clone());
            cv::detail::waveCorrect(rmats, wave_correct_kind_);
            for (size_t i = 0; i < cameras_.size(); ++i)
                cameras_[i].R = rmats[i];
        }
        
        return OK;
    }
    
    
    cv::Ptr<SRStitcher> createStitcher(bool try_use_gpu)
    {
        cv::Ptr<SRStitcher> stitcher = cv::makePtr<SRStitcher>();
        stitcher->setRegistrationResol(0.6);
        stitcher->setSeamEstimationResol(0.1);
        stitcher->setCompositingResol(SRStitcher::ORIG_RESOL);
        stitcher->setPanoConfidenceThresh(1);
        stitcher->setWaveCorrection(true);
        stitcher->setWaveCorrectKind(cv::detail::WAVE_CORRECT_HORIZ);
        stitcher->setFeaturesMatcher(cv::makePtr<cv::detail::BestOf2NearestMatcher>(try_use_gpu));
        stitcher->setBundleAdjuster(cv::makePtr<cv::detail::BundleAdjusterRay>());
        
#ifdef HAVE_CUDA
        if (try_use_gpu && cuda::getCudaEnabledDeviceCount() > 0)
        {
#ifdef HAVE_OPENCV_NONFREE
            stitcher->setFeaturesFinder(makePtr<detail::SurfFeaturesFinderGpu>());
#else
            stitcher->setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
            stitcher->setWarper(makePtr<SphericalWarperGpu>());
            stitcher->setSeamFinder(makePtr<detail::GraphCutSeamFinderGpu>());
        }
        else
#endif
        {
#ifdef HAVE_OPENCV_NONFREE
            stitcher->setFeaturesFinder(cv::makePtr<cv::detail::SurfFeaturesFinder>());
#else
            stitcher->setFeaturesFinder(cv::makePtr<cv::detail::OrbFeaturesFinder>());
#endif
            stitcher->setWarper(cv::makePtr<cv::SphericalWarper>());
            stitcher->setSeamFinder(cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR));
        }
        
        stitcher->setExposureCompensator(cv::makePtr<cv::detail::BlocksGainCompensator>());
        stitcher->setBlender(cv::makePtr<cv::detail::MultiBandBlender>(try_use_gpu));
        
        return stitcher;
    }
} // namespace cv

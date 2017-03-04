//
//  convertTool.h
//  Stitcher
//
//  Created by a on 2016/2/19.
//  Copyright © 2017年 a. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface convertTool : NSObject

+(UIImage *)imageFromMat:(const cv::UMat&)cvMat;
+(cv::Mat)CVMat:(UIImage *)img;
@end

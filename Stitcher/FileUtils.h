//
//  FileUtils.h
//  Stitcher
//
//  Created by a on 2016/2/20.
//  Copyright © 2017年 a. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface FileUtils : NSObject

+(NSString *)connersPath;
+(NSString *)docPath;
+(NSString *)sizesPath;
+(NSString *)warpImagePath:(size_t)index;
+(NSString *)maskImagePath:(size_t)index;
+(NSString *)blendImagePath:(size_t)index;

@end

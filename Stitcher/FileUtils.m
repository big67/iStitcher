//
//  FileUtils.m
//  Stitcher
//
//  Created by a on 2016/2/20.
//  Copyright © 2017年 a. All rights reserved.
//

#import "FileUtils.h"

@implementation FileUtils

+(NSString *)docPath
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    return [paths objectAtIndex:0];
}

+(NSString *)connersPath
{
    return [NSString stringWithFormat:@"%@/panoTemp/conners.archive", [self docPath]];
}

+(NSString *)sizesPath
{
    return [NSString stringWithFormat:@"%@/panoTemp/sizes.archive", [self docPath]];
}

+(NSString *)warpImagePath:(size_t)index
{
    return [NSString stringWithFormat:@"%@/panoTemp/image_%zu.png", [self docPath], index];
}

+(NSString *)maskImagePath:(size_t)index
{
    return [NSString stringWithFormat:@"%@/panoTemp/mask_%zu.png", [self docPath], index];
}

+(NSString *)blendImagePath:(size_t)index
{
    return [NSString stringWithFormat:@"%@/panoTemp/blend_%zu.png", [self docPath], index];
}

@end

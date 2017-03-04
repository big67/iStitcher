//
//  ViewController.m
//  Stitcher
//
//  Created by a on 2016/2/19.
//  Copyright © 2017年 a. All rights reserved.
//

#import "ViewController.h"
#import "UIImage+OpenCV.h"
#import "FileUtils.h"
#import "convertTool.h"

size_t warpImageSize;


@interface ViewController ()
{
}
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.

    NSLog(@"create temporary directory");
    
    NSFileManager *fm =[NSFileManager defaultManager];
    [fm createDirectoryAtPath:[NSString stringWithFormat:@"%@/panoTemp", [FileUtils docPath]] withIntermediateDirectories:YES attributes:nil error:nil];
    
    NSLog(@"warpping...");
    
    NSMutableArray *imageNames = [NSMutableArray array];
    for(int i=0; i<5; i++){
        [imageNames addObject:[NSString stringWithFormat:@"%d", i]];
    }
    
    [self warpImages:imageNames];
    
    NSLog(@"blending...");
    NSMutableArray *sources = [NSMutableArray array];
    NSMutableArray *masks   = [NSMutableArray array];
    for(int i=0; i<warpImageSize; i++){
        [sources addObject:[FileUtils warpImagePath:i]];
        [masks addObject:[FileUtils maskImagePath:i]];
    }
    [self blendImages:sources maskNames:masks];
    
    [sources removeAllObjects];
    [masks removeAllObjects];
    
    NSLog(@"composing...");
    NSArray *conners = [NSKeyedUnarchiver unarchiveObjectWithFile:[FileUtils connersPath]];
    NSArray *sizes = [NSKeyedUnarchiver unarchiveObjectWithFile:[FileUtils sizesPath]];
    
    NSMutableArray *blendedimageName   = [NSMutableArray array];
    for(int i=0; i<warpImageSize; i++){
        [blendedimageName insertObject:[FileUtils blendImagePath:i] atIndex:i];
    }
    UIImage *finalImage;
    if(warpImageSize>0){
        finalImage = [self composeImages:blendedimageName points:conners sizes:sizes];
    }

    
    NSLog(@"clean up");
    [fm removeItemAtPath:[NSString stringWithFormat:@"%@/panoTemp", [FileUtils docPath]] error:nil];
    
    NSLog(@"finish");
    
}


-(UIImage *)composeImages:(NSArray *)imagesName points:(NSArray *)points sizes:(NSArray *)sizes
{
    CGFloat minx, maxx;
    CGFloat miny, maxy;
    CGFloat minW, maxW;
    CGFloat minH, maxH;
    
    CGPoint firstPoint = [points[0] CGPointValue];
    minx = maxx = firstPoint.x;
    maxy = miny = firstPoint.y;
    minH = maxW = minW = maxH = 0;
    int lastIndex = 0;
    
    for(int index=0; index<points.count; index++){
        CGPoint mp = [points[index]CGPointValue];
        CGSize size = [sizes[index]CGSizeValue];
        
        minx = fmin(mp.x, minx);
        maxx = fmax(mp.x, maxx);
        if(maxx == mp.x){
            lastIndex = index;
        }
        
        
        miny = fmin(mp.y, miny);
        maxy = fmax(mp.y, maxy);
        
        minW = fmin(minW, size.width);
        minH = fmin(minH, size.height);
        
        maxW = fmax(maxW, size.width);
        maxH = fmax(maxH, size.height);
    }
    
    CGFloat markWidth = maxx-minx+[sizes[lastIndex]CGSizeValue].width;
    CGFloat height = maxH;
    CGFloat width = fmax(markWidth, maxW);
    
    CGSize size = CGSizeMake(width, height);
    
    UIGraphicsBeginImageContext(size);
    
    for(int index=0; index<points.count; index++){
        
        UIImage *img = [UIImage imageWithContentsOfFile:imagesName[index]];
        CGPoint drawPoint = [points[index]CGPointValue];
        CGSize size = [sizes[index]CGSizeValue];
        
        [img drawInRect:CGRectMake(drawPoint.x-minx, drawPoint.y-firstPoint.y, size.width, size.height)];
    }
    
    UIImage *finalImage = UIGraphicsGetImageFromCurrentImageContext();
    
    UIGraphicsEndImageContext();
    
    return finalImage;
}

-(void)blendImages:(NSArray *)srcName maskNames:(NSArray *)maskNames
{
    for(int index=0; index<maskNames.count; index++){
//        [self blend1:srcName mask:maskNames index:index];
        [self blend:srcName mask:maskNames index:index];
    }
 
}

-(void)blend:(NSArray *)srcName mask:(NSArray *)maskNames index:(int)index
{
    @autoreleasepool {
        UIImage *overlayImage = [UIImage imageWithContentsOfFile:srcName[index]];
        UIImage *backgroundImage = [UIImage imageWithContentsOfFile:maskNames[index]];
        
        CIImage *moi2 = [CIImage imageWithCGImage:overlayImage.CGImage];
        CIImage *gradimage = [CIImage imageWithCGImage:backgroundImage.CGImage];
        
        CIFilter* blend = [CIFilter filterWithName:@"CIBlendWithMask"];
        [blend setValue:moi2 forKey:@"inputImage"];
        [blend setValue:gradimage forKey:@"inputMaskImage"];
        
        CIContext *context = [CIContext contextWithOptions:nil];
        CIImage *outputImage = blend.outputImage;
        
        CGImageRef image = [context createCGImage:outputImage fromRect:outputImage.extent];
        UIImage *blendedImage = [UIImage imageWithCGImage:image];
        CGImageRelease(image);
        
        NSData *imageData = UIImagePNGRepresentation(blendedImage);
        [imageData writeToFile:[FileUtils blendImagePath:index] atomically:YES];
    }

}



-(void)warpImages:(NSArray *)imgNames
{
    std::vector<cv::Mat> sources;
    
    sr::SRStitcher stitcher = sr::SRStitcher::createDefault(false);
    
    for(int i=0; i<imgNames.count; i++){
        @autoreleasepool {
            UIImage *srcImage = [UIImage imageNamed:imgNames[i]];
            cv::Mat mat = [srcImage CVMat3];
//            cv::Mat mat = [convertTool CVMat:srcImage];
            sources.push_back(mat);
        }
    }
    
    sr::SRStitcher::Status status = stitcher.getWrapImageAndMask(sources, warpImageSize);
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end

function [avgPSNR, SSIM] = computescores(Refpath,Outpath,Inpath)
ref_files = dir(Outpath);
PSNR = [];SSIMscores= [];
for i = 3:numel(ref_files)
    imname = ref_files(i).name;
    Ref_image = imread([Refpath,'/',imname]);
    Out_image = imread([Outpath,'/',imname]);
    In_image = imread([Inpath,'/',imname]); In_image = imresize(In_image,4);
    
    PSNRscores(i-2,1) = psnr(In_image, Ref_image); 
    PSNRscores(i-2,2) = psnr(Out_image, Ref_image);
    
    [ssimval1, ssimmap] = ssim(In_image, Ref_image);
    [ssimval2, ssimmap] = ssim(Out_image, Ref_image);
    SSIMscores(i-2,1) = ssimval1;
    SSIMscores(i-2,2) = ssimval2;
end 
avgPSNR = mean(PSNRscores); 
avgSSIM = mean(SSIMscores); 
end 

%%% for multiple level of compression

function [avgPSNR, SSIM] = computescores(Refpath,Outpath,Inpath)
ref_files = dir(Refpath);
PSNR = [];SSIMscores= [];
for i = 3:numel(ref_files)
    imname = ref_files(i).name; [filepath,name,ext] = fileparts(imname);
    Ref_image = imread([Refpath,'/',imname]);
    A = imread([Outpath,'/',name, '.png']);
    Out_image = imresize(A,8 );
%    In_image = imread([Inpath,'/',name, '_quality_181.png']); 
    In_image = imresize(A,8,'nearest');
    PSNRscores(i-2,1) = psnr(In_image, Ref_image); 
    PSNRscores(i-2,2) = psnr(Out_image, Ref_image);
    
    [ssimval1, ssimmap] = ssim(In_image, Ref_image);
    [ssimval2, ssimmap] = ssim(Out_image, Ref_image);
    SSIMscores(i-2,1) = ssimval1;
    SSIMscores(i-2,2) = ssimval2;
end 
avgPSNR = mean(PSNRscores); 
avgSSIM = mean(SSIMscores); 
end 




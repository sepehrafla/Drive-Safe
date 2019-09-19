function imageArray = imageProcessing(filePath)
    more off;
    Files = dir(filePath);
    images = [];
    for i = 1:3
      dirName = Files(i).name;
      if(regexp(dirName, '^\w+'))
          dirPath = strcat(filePath, '/', dirName);
          ImageFiles = dir(dirPath);
          for j = 3:500
              imageName = ImageFiles(j).name;
              imgPath = strcat(dirPath, '/', imageName);
              RGB = imread(imgPath);
              RGBResize = imresize(RGB, 0.40);
              BW = rgb2gray(RGBResize);
              BW = double(BW);
              images = [images; BW(:)'];
              disp(j)
              end
              MatFilePath = strcat(filePath, '/' , strcat(dirName, int2str(1), '.mat'));
              save(MatFilePath, 'images');
              images = [];
         end
      end
    images;
end

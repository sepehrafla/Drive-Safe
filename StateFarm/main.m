pkg load image;
more off;

trainImagesPath = 'C:/Users/aravi/Documents/Aravind/StateFarmNeuralNetwork/StateFarm/Images/train';
dataPath = 'C:/Users/aravi/Documents/Aravind/StateFarmNeuralNetwork/StateFarm/Data'
% Process Images
printf('Processing Images...\n')

images = imageProcessing(trainImagesPath);

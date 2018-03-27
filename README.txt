================================================================================
Database of Human Attributes (HATDB)
https://users.info.unicaen.fr/~gsharma/hatdb/
================================================================================

This file describes the use of HATDB (Database of human attributes) as 
introduced in

[1] G. Sharma and F. Jurie, Learning discriminative spatial representation for 
    image classification, British Machine Vision Conference, 2011

Kindly cite [1] if you use this dataset in your research.

For questions or comments, kindly send a mail to
gaurav [dot] sharma [at] {unicaen or inria} [dot] fr

================================================================================

The images were downloaded from Flickr.com and most of them have a Creative 
Commons license. The images are the property of their respective owners and
the use must respect the terms of use layed down by Flickr.com at
http://www.flickr.com/help/terms/

If you are the owner of any photo in the dataset and would like us to take
off your photo, kindly send a mail to the authors.

================================================================================

The archive contains a folder with images, the annotation as a MATLAB mat file 
<anno.mat> with a MATLAB structure:

anno =

     classes: {27x1 cell}
       files: {1x9344 cell}
      objbbs: [9344x5 double]
      objids: [1x9344 double]
           y: [27x9344 double]
    trainidx: [1x3500 double]
      validx: [1x3500 double]
     testidx: [1x2344 double]

* The attributes are named in <anno.classes>
* The i-th human is in image <anno.files{i}> and has a bounding box 
  <anno.objbbs(i,:)> (x1,y1,x2,y2) and ID <anno.objids(i)>. The annotation
  for the 27 attributes for the i-th human is in <anno.y(:,i)>.
* The object ID is for differentiating multiple humans in the same image.
  (anno.files{i},anno.objids(i)) is the unique identifier of a human in the DB.
* Thus, k-th row of <anno.y> corresponds to the <anno.classes{k}> attribute
* The possible values for annotations <anno.y> are 
        +1 for attribute present
        -1 for attribute absent
         0 for attribute not visible or ambiguous
* The train/val/testidx contain the indices of the humans used for training
  validation and testing in [1].
================================================================================

Pseudo code (MATLAB like) for training a classifier for *ATTRIBUTE k* will be

% Features for all humans
for human=1:length(anno.files)
    I = imread([path_to_hatdb/images/' anno.files{human}]);
    bb = anno.objbbs(human, :);
    humanI = imcrop(I, [bb(1) bb(2) bb(3)-bb(1) bb(4)-bb(2)]);
    % imshow(humanI); 
    % title(['human #' human ' in img:' anno.files{human} ', id:' int2str(anno.objids(human)]);
    features(:,human) = get_features(humanI);
end

% Attribute k 
y        = anno.y(k,:)';
trainidx = anno.trainidx;
validx   = anno.validx;
testidx  = anno.testidx;

% Remove the not present/ambiguous examples 
trainidx = trainidx(y(trainidx)~=0);
validx   = validx(y(validx)~=0);
testidx  = testidx(y(testidx)~=0);

% Annotations for train/val/test sets 
ytrain   = y(trainidx);
yval     = y(validx);
ytest    = y(testidx);

% Features for train/val/test sets 
train_features = features(:, trainidx);
val_features   = features(:, validx);
test_features  = features(:, testidx);

% Do cross validation, train model and report prediction result on test set
best_params = do_crossvalidation (train_features, ytrain, val_features, yval);
model       = train_model([train_features val_features], [ytrain; yval], best_params);
prediction  = test_model(test_features, model);
result      = calc_AP (prediction, ytest);

**** EOF ****

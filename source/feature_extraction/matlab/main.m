%change the following directories if needed
%Implementation of the method of Bulacu et al. :
%Writer Identification Using Edge-Based Directional Features
%Author: Abdelâali Hassaïne
%Date: February 2012
train_rep='D:\awic2012\images\training\'
test_rep='D:\awic2012\images\test\'
%list of training images
train_images=dir(train_rep);
%trying several number of directions
for hist_size=4:4:16
    %creating output file for each size
    output_file=['D:\awic2012\hist_' num2str(hist_size) '.csv'];
    fid = fopen(output_file, 'w');
    fprintf(fid,'image_ID,writer\n');
    hists=[];
    names=[];
    disp('Computing distribution features for training set...\n');
    % 1st and 2nd file being '.' and '..'
    for j=3:length(train_images)
        fprintf(['Processing training image ' train_images(j).name '...\n']);
        im_candidate=imread([train_rep train_images(j).name]);
        %computing directions histogram
        hist=hist_directions(im_candidate,hist_size);
        hists=[hists;hist];
        [pathstr, name, ext] = fileparts(train_images(j).name);
        names=[names;name];
    end
    for i=3:length(test_images)
        fprintf(['Processing test image ' test_images(i).name '...\n']);
        im_questioned=imread([test_rep test_images(i).name]);
        %computing directions histogram
        hist=hist_directions(im_questioned,hist_size);
        mindistance=bitmax;
        %comparing with all the training images and keeping the closest
        for j=1:length(names)
            diff=hist-hists(j,:);
            distance=sum(diff.*diff);
            if(distance<mindistance)
                mindistance=distance;
                best_writer=names(j,:);
                %writer ID is in the first three characters
                best_writer=best_writer(1:3);
            end
        end
        [pathstr, name, ext] = fileparts([test_rep test_images(i).name]);
        fprintf(fid, '%d,%d\n',str2double(name),str2double(best_writer));
    end
end   
function hist=  hist_directions(image,n)
if(n==4)
    hist=hist_4directions(image);
elseif(n==8)
    hist=hist_8directions(image);
elseif(n==12)
    hist=hist_12directions(image);
elseif(n==16)
    hist=hist_16directions(image);
end
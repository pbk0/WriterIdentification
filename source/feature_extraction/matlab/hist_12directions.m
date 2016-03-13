function hist=  hist_12directions(image)
hist=zeros(1,12);
nb=0;
edge1=edge(image);
width=size(image,1);
height=size(image,2);
for x=1:width
    for y=1:height
        if(edge1(x,y))
            for i=0:11
                myX=x+xC3(i);
                myY=y+yC3(i);
                if((myX>=1)&&(myX<=width)&&(myY>=1)&&(myY<=height))
                    if(edge1(myX,myY))
                        hist(i+1)=hist(i+1)+1;
                        nb=nb+1;
                    end
                end
            end
        end
    end
end
hist=hist/nb;
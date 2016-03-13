function y = yC2(i)
	if(i==0)y= 0;
    elseif(i==1)y= -1;
    elseif(i==2)y= -2;
    elseif(i==3)y= -2;
    elseif(i==4)y= -2;
    elseif(i==5)y= -2;
    elseif(i==6)y= -2;
    elseif(i==7)y= -1;
	else y= yC2(i-8);
    end
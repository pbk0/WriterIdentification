function x = xC2(i)
	if(i==0)x= 2;
	elseif(i==1)x= 2;
	elseif(i==2)x= 2;
	elseif(i==3)x= 1;
	elseif(i==4)x= 0;
	elseif(i==5)x= -1;
	elseif(i==6)x= -2;
	elseif(i==7)x= -2;
	else x= xC2(i-8);
    end
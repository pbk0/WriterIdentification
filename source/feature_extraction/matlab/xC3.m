function x = xC3(i)
	if(i==0)x= 3;
	elseif(i==1)x= 3;
	elseif(i==2)x= 3;
	elseif(i==3)x= 3;
	elseif(i==4)x= 2;
	elseif(i==5)x= 1;
	elseif(i==6)x= 0;
	elseif(i==7)x= -1;
	elseif(i==8)x= -2;
	elseif(i==9)x= -3;
	elseif(i==10)x= -3;
	elseif(i==11)x= -3;
	else x= xC3(i-12);
    end
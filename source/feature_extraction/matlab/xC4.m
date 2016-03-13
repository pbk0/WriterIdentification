function x = xC4(i)
	if(i==0)x= 4;
	elseif(i==1)x= 4;
	elseif(i==2)x= 4;
	elseif(i==3)x= 4;
	elseif(i==4)x= 4;
	elseif(i==5)x= 3;
	elseif(i==6)x= 2;
	elseif(i==7)x= 1;
	elseif(i==8)x= 0;
	elseif(i==9)x= -1;
	elseif(i==10)x= -2;
	elseif(i==11)x= -3;
	elseif(i==12)x= -4;
	elseif(i==13)x= -4;
	elseif(i==14)x= -4;
	elseif(i==15)x= -4;
	else x= xC4(i-16);
    end
a=[0 1 0]

y=[1  4  1;
   2  3  1;
  -4 -1 -1;
  -3 -2 -1];
while 1
    mistake = [0,0,0];
    for i = 1:4
        if a * (y(i,:))' < 0
            mistake = mistake + y(i,:);
        end
    end
    if mistake == [0 0 0]
        break
    else
    a = a + mistake;
    end
end
a
r = [0.17  0.25  0.21  0.16  0.07  0.08  0.04  0.02];
length = size(r);
T = zeros(length);
T(1) = r(1);
for i = 2:length(2)
    T(i) = T(i-1) + r(i);
end
T = T * (length(2) - 1);
T = round(T);

s = zeros(length);
for i =1:length(2)
    s(T(i)+1) = s(T(i)+1) + r(i);
end
fig = figure(1);
bar(0:length(2)-1, s);
title("变换后的直方图");
xlabel("$s_k$", "Interpreter","latex")
ylabel("$p\left(s_k\right)$", "Interpreter","latex")
savePath = fullfile("..", "result", "变换后的直方图示意图.jpg");
saveas(fig, savePath)
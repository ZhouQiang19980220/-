clf;
r = -0:0.01:1;
m = max(r)/4;
E = 5:5:20;
s = zeros(size(r,2),size(E,2));
str_E = strings(size(E));
fig = figure(1);
hold on;
for i =1:size(E,2)
    str_E(i) = strcat("E = ",num2str(E(i)));
    s(:,i) = 1 ./ (1 + (m ./ (r + eps)).^E(i));
    plot(r,s);
end
hold off;
title('$$T(r)=\frac{1}{1+\left(\frac{m}{r}\right)^{E}}$$','Interpreter','latex')
xlabel('$$r$$','Interpreter','latex')
ylabel('$$T(r)$$','Interpreter','latex')
legend(str_E)
savePath = fullfile('..','result','..\result\不同斜率的对比度拉伸函数.jpg');
saveas(fig,savePath);
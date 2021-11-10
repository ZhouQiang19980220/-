clc
clear
close all

x1 = 0.5*ones(100,1);
y12 = linspace(0.25,10,100);

x2 = linspace(-10,0.5,100);
y13 = 0.5 * x2;

x3 = linspace(0.5,10,100);
y23 = 0.5 * (1-x3);

w1 = and(x1 < 0.5, y13 > 0.5 * x2);
hold on
plot(x1,y12)
plot(x2,y13)
plot(x3,y23)
text([-3; 0.5;3],[0.5; -2;0.5],{'R1';'R3';'R2'})
hold off
xlim([-5,5])
ylim([-5,5])
legend('w1和w2决策面','w1和w3决策面','w2和w3决策面')
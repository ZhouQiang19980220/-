imgPath = fullfile("../", "source/", "1_3_1.tif");
img = imread(imgPath);

fig = figure(1);
subplot(3,2,1); imshow(img); title("原图");
subplot(3,2,2); imhist(img); title("原图的直方图");

img_1 = histeq(img);
subplot(3,2,3); imshow(img_1); title("第一次直方图均衡");
subplot(3,2,4); imhist(img_1); title("第一次直方图均衡后的直方图");

img_2 = histeq(img_1);
subplot(3,2,5); imshow(img_2); title("第二次直方图均衡");
subplot(3,2,6); imhist(img_2); title("第二次直方图均衡后的直方图");

suptitle("多次直方图均衡效果对比")

% 保存图像
savePath = fullfile("../", "result/", "多次直方图均衡效果对比.jpg");
saveas(fig, savePath)
imgPath = fullfile("../", "source/", "1_3_1.tif");
img = imread(imgPath);

fig = figure(1);
subplot(3,2,1); imshow(img); title("ԭͼ");
subplot(3,2,2); imhist(img); title("ԭͼ��ֱ��ͼ");

img_1 = histeq(img);
subplot(3,2,3); imshow(img_1); title("��һ��ֱ��ͼ����");
subplot(3,2,4); imhist(img_1); title("��һ��ֱ��ͼ������ֱ��ͼ");

img_2 = histeq(img_1);
subplot(3,2,5); imshow(img_2); title("�ڶ���ֱ��ͼ����");
subplot(3,2,6); imhist(img_2); title("�ڶ���ֱ��ͼ������ֱ��ͼ");

suptitle("���ֱ��ͼ����Ч���Ա�")

% ����ͼ��
savePath = fullfile("../", "result/", "���ֱ��ͼ����Ч���Ա�.jpg");
saveas(fig, savePath)
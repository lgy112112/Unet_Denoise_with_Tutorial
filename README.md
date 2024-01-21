# 欢迎来到我的仓库

这个仓库使用Unet实现了降噪任务。为了方便大家操作，我在`unet_denoise_tutorial.ipynb`文件里详细写出了每一步。

只要执行这个文件，就可以完成数据库的导入、分类等所有操作。需要注意的是，我在这里没有对训练集和测试集进行分类。

我非常建议你用Google Colab运行这个notebook文件，但在本地运行也没问题，除了配置环境有少许困难。

如果你有任何建议或者遇到困难，请及时提出，我会非常感激。

请多多支持最近回归的韩国女子团体NMIXX！

![image](https://github.com/lgy112112/Unet_Denoise_with_Tutorial/assets/144128974/f4858b47-e072-4b21-a68b-18123950c6c9)

在高斯噪声sigma=80情况下表现优异。Matlab下测得数据，MSE似乎有些异常，带噪图像MSE应为80左右。
| 图片类型         | MSE       | PSNR    | SSIM     |
|-----------------|-----------|---------|----------|
| 原始图片         | 287.276   | 19.482  | 0.654    |
| Mean Filtered   | 258.848   | 19.881  | 0.772    |
| Median Filtered | 262.214   | 19.841  | 0.761    |
| BM3D Filtered   | 248.767   |20.156   |0.670     |
| DnCNN Denoised  |255.025    |24.28	  |0.771	   |
| Unet Denoised   |14.22      |36.60	  |0.901	   |




# Welcome to my repository



This repository implements a denoising task using Unet. For your convenience, I have detailed each step in the `unet_denoise.ipynb` file. 

By running this file, you can complete all operations such as database import and classification. 

Please note that I did not classify the training set and test set here.

Google Colab is highly recommended to run this notebook. Local is ok but may meet some problems setting environment.

If you have any suggestions or difficulties, please feel free to raise them, I would be very grateful.

Please support the recently returned Korean girl group NMIXX!

![image](https://github.com/lgy112112/Unet_Denoise_with_Tutorial/assets/144128974/35f215f4-ad9d-43ca-bdf8-c2784abad69f)


The performance is excellent under Gaussian noise with sigma=80. The data measured in Matlab, MSE seems to be a bit abnormal, the MSE of the noisy image should be around 80.

| Image Type       | MSE       | PSNR    | SSIM     |
|-----------------|-----------|---------|----------|
| Original Image  | 287.276   | 19.482  | 0.654    |
| Mean Filtered   | 258.848   | 19.881  | 0.772    |
| Median Filtered | 262.214   | 19.841  | 0.761    |
| BM3D Filtered   | 248.767   |20.156   |0.670     |
| DnCNN Denoised  | 255.025   |24.28	  |0.771	   |
| Unet Denoised   | 14.22     |36.60	  |0.901	   |

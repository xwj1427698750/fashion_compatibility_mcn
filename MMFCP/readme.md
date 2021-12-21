#文件内容介绍：
- generator_show.py 展示测试集中，根据多件单品生成的一件单品的图像, 新建一个目录generator_show，每一个套装存储在一个文件夹里，given开头的图片表示
    输入的3件单品，low_rec.jpg和high_rec.jpg表示生成的图片，high_rec.jpg比low_rec.jpg更清晰，option开头的图片是fitb任务给出的
    选项，target.jpg是目标生成的图片。true.txt表示fitb任务选择正确，false.txt表示选择错误, txt文件内容是正负样本的diff_score得分。
- model.py 包含网络结构模型
- polyvore_dataset.py  构建获取数据的dataset类
- train.py 训练脚本
- evaluate.py 测试脚本
- utils.py 包含：保存最优模型结果的函数， 构建dataLoader的函数，日志相关

# 指令
## 输出搭配生成可视化结果的
CUDA_VISIBLE_DEVICES=0 python generator_show.py --model_path="./MMFCP_atten_head(4)_feature_size(96)_input_off(F)_generator_off(F)_mlmsff_off(F)_acc.pth" 
## 自定义单项属性的命令
### 自定义模型存储名称示例
训练：CUDA_VISIBLE_DEVICES=0 python train.py --comment="XXX"  
测试：CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./XXX_acc.pth"  

## 论文实验中各项结果对应的指令
### 最优效果的命令：

### 对比实验：多层级特征维度选择
- feature_size = 64   实验室服务器 cuda:0
训练：CUDA_VISIBLE_DEVICES=0 python train.py --comment="MMFCP_atten_head(4)_feature_size(64)_input_off(F)_generator_off(F)_mlmsff_off(F)" --feature_size=64
测试：CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MMFCP_atten_head(4)_feature_size(64)_input_off(F)_generator_off(F)_mlmsff_off(F)_acc.pth" --feature_size=64  
fitb@4: 0.8040   fitb@5: 0.7577   fitb@6: 0.7211
 
- feature_size = 96   实验室服务器 cuda:1
训练：CUDA_VISIBLE_DEVICES=1 python train.py --comment="MMFCP_atten_head(4)_feature_size(96)_input_off(F)_generator_off(F)_mlmsff_off(F)" --feature_size=96
测试：CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_path="./MMFCP_atten_head(4)_feature_size(96)_input_off(F)_generator_off(F)_mlmsff_off(F)_acc.pth" --feature_size=96  
fitb@4: 0.7986   fitb@5: 0.7523   fitb@6: 0.7269

- feature_size = 128  实验室部署服务器 cuda:0
训练：CUDA_VISIBLE_DEVICES=0 python3 train.py --comment="MMFCP_atten_head(4)_feature_size(128)_input_off(F)_generator_off(F)_mlmsff_off(F)" --feature_size=128
测试：CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --model_path="./MMFCP_atten_head(4)_feature_size(128)_input_off(F)_generator_off(F)_mlmsff_off(F)_acc.pth" --feature_size=128  
fitb@4: 0.8119   fitb@5: 0.7598   fitb@6: 0.7282

- feature_size = 256  实验室部署服务器 haibin:0
训练：CUDA_VISIBLE_DEVICES=0 python3 train.py --comment="MMFCP_atten_head(4)_feature_size(256)_input_off(F)_generator_off(F)_mlmsff_off(F)" --feature_size=256
测试：CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --model_path="./MMFCP_atten_head(4)_feature_size(256)_input_off(F)_generator_off(F)_mlmsff_off(F)_acc.pth" --feature_size=256
fitb@4:    fitb@5:    fitb@6: 


### 消融实验：完全移除多层级特征模块

### 消融实验：移除多层级特征模块中的某些层
     
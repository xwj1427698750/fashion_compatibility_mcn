#文件内容介绍：
- compat show.py 
将测试集的测试结果按照TP（搭配的套装被正确预测），TN（搭配的套装被错误预测）
，FP（不搭配的套装被正确预测），FN（不搭配的套装被错误预测）这四类分别存
- model.py 包含网络结构模型
- polyvore_dataset.py  构建获取数据的dataset类
- resnet.py 残差神经网络结构
- train.py 训练脚本
- evaluate.py 测试脚本
- utils.py 包含：保存最优模型结果的函数， 构建dataLoader的函数，日志相关

# 指令
## 输出搭配可视化结果的
CUDA_VISIBLE_DEVICES=0 python compat show.py --model_path="./MLMSFF_layer_size(256)_multi_layer(4)_auc.pth" --layer_size=256
## 自定义单项属性的命令
### 自定义模型存储名称示例
训练：CUDA_VISIBLE_DEVICES=0 python train.py --comment="XXX"  
测试：  
采用auc指标：CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./XXX_auc.pth"  
采用acc指标：CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./XXX_acc.pth"  
后续采用auc指标示例

### 自定义多层级特征融合模块中的层级特征维度示例
训练：CUDA_VISIBLE_DEVICES=0 python train.py --comment="XXX" --layer_size=256   
测试：CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./XXX_auc.pth" --layer_size=256

### 自定义多层级特征融合模块中使用的层级特征数示例
训练：CUDA_VISIBLE_DEVICES=0 python train.py --comment="XXX" --multi_layer=3      
测试：CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./XXX_auc.pth" --multi_layer=3

### 是否取消VSE模块示例
训练：CUDA_VISIBLE_DEVICES=0 python train.py --comment="XXX" --vse_off=TRUE    
测试：测试的时候不需要VSE模块，采用默认的FALSE就可以  
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./XXX_auc.pth"

## 论文实验中各项结果对应的指令
### 最优效果的命令：
训练：CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(256)_multi_layer(4)" --layer_size=256      
测试：CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(256)_multi_layer(4)_auc.pth" --layer_size=256  
auc： 0.9005  fitb@4  acc is   0.6300
### 对比实验：多层级特征维度选择
- layer_size = 64  
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(64)_multi_layer(4)" --layer_size=64      
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(64)_multi_layer(4)_auc.pth" --layer_size=64      
auc： 0.8928   fitb@4  acc is  0.6149

- layer_size = 128  
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(128)_multi_layer(4)" --layer_size=128  
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(128)_multi_layer(4)_auc.pth" --layer_size=128
auc： 0.8942   fitb@4  acc is  0.6259

- layer_size = 256  
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(256)_multi_layer(4)" --layer_size=256    
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(256)_multi_layer(4)_auc.pth" --layer_size=256
auc： 0.9005  fitb@4  acc is   0.6300

- layer_size = 512  
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(512)_multi_layer(4)" --layer_size=512  
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(512)_multi_layer(4)_auc.pth" --layer_size=512
auc： 0.8893   fitb@4  acc is  0.6248

### 消融实验：完全移除多层级特征模块
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(256)_multi_layer(0)" --multi_layer=0  
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(256)_multi_layer(0)_auc.pth" --multi_layer=0 

### 消融实验：移除多层级特征模块中的某些层
- 移除第4层  
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(256)_multi_layer(3)" --layer_size=256 --multi_layer=3  
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(256)_multi_layer(3)_auc.pth" --layer_size=256 --multi_layer=3

- 移除第3、4层  
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(256)_multi_layer(2)" --layer_size=256 --multi_layer=2  
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(256)_multi_layer(2)_auc.pth" --layer_size=256 --multi_layer=2  

- 移除第2、3、4层  
训练： CUDA_VISIBLE_DEVICES=0 python train.py --comment="MLMSFF_layer_size(256)_multi_layer(1)" --layer_size=256 --multi_layer=1  
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_path="./MLMSFF_layer_size(256)_multi_layer(1)_auc.pth" --layer_size=256 --multi_layer=1     
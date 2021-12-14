##文件内容介绍：
- compat show.py 
将测试集的测试结果按照TP（搭配的套装被正确预测），TN（搭配的套装被错误预测）
，FP（不搭配的套装被正确预测），FN（不搭配的套装被错误预测）这四类分别存
- model.py 包含网络结构模型
- polyvore_dataset.py  构建获取数据的dataset类
- resnet.py 残差神经网络结构
- train.py 训练脚本
- evaluate.py 测试脚本
- utils.py 包含：保存最优模型结果的函数， 构建dataLoader的函数，日志相关

## 指令
训练： CUDA_VISIBLE_DEVICES=0 python train.py
测试： CUDA_VISIBLE_DEVICES=0 python evaluate.py
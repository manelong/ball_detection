https://github.com/nttcom/WASB-SBDT/

# 流程计划

1. 配置环境

   - ```shell
     pip install numpy==1.22.4
     conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
     pip install hydra-core==1.2.0 tqdm==4.64.0 opencv-python==4.6.0.66 scikit-learn==1.1.1 scikit-image==0.19.3 pandas==1.3.5 einops==0.4.1 timm==0.6.5 matplotlib==3.5.2
     ```

2. 熟悉代码框架

   - 跑通repo中给的demo：WASB (Ours, Step=1) - Volleyball ✅
   - 由于volleyball_数据集太大，跑通demo时仅选择部分testing样本上传到服务器。对此，需要进行修改：1、src/configs/dataset/volleyball.yaml中的root_dir和frame_dirname修改为对应路径；2、src/configs/dataset/volleyball.yaml中的test.matches修改为[4]，对应数据集videos_mini ✅

3. 理解数据读取代码 ✅

4. 理解微调代码 ✅

5. 准备项目训练数据

   - 使用repo提供的参数验证项目数据集的效果✅==效果很差==
   - 查看源代码是否有限制只检测一个球的操作✅（初步推测只有tracker会执行该操作）
   - 检查sampler原理✅（没问题，是保证了3张连续的样本）
   - 跑通单球训练代码✅
   - 整理训练数据格式✅

6. 使用项目训练数据进行微调✅

7. 分析结果

   - 分析损失收敛情况：demo_data-V2-batch1_single_ball_anno0~10代损失下降正常，但是10~20代损失下降很慢。demo_data-V2-batch1_single_ball_anno_continueOn10epochsFor20

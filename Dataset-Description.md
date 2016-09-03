# VQA dataset整理

* COCO-QA (Real image)
    1. 数据集描述： 来源于Virginia Tech里的VQA team。图片来自于微软（COCO）数据集， 问题和答案有部分从COCO image descriptions归类后得到，按答案分为object，number，color，location四类，后来加入了YES/NO一类。后经人工标注包含了使得问题集更为广泛和有趣。经过评估和调查，18%的问题需要常识判断，5.5%的问题需要adult-level的知识，
    2. 数据集[地址](http://visualqa.org/download.html)
    3. 图片数量：82,783(training); 40,504(validation)
    4. 问题数量：2,483,49(training); 1,215,12(validation)
    5. 答案数量：2,483,490(training); 1,215,120(validation)

* DAQUAR
    1. 数据集描述： 这是第一个作为VQA benchmark的数据集。 图片来源于NYU—Depth v2 dataset, 问题和答案部分来源于NYU数据集模板自动生成，部分来源于人工标注。这个数据集是基于 **indoor** 环境中的（跟我们的项目相对口），缺点是它的问答聚焦在16种颜色和894中物体（大部分是显眼的比如table，chair等）
    2. 数据集[地址]()
    3. 图片数量：795(training); 654 (test)
    4. 问题数量：6794(training); 5674 (testing)
    5. 答案数量：6794(training); 5674 (testing)

* FM-IQA 
    1. 数据集描述： 百度的数据集。图片也是来自于COCO，但是不同的是它的问答全部是由人工（借助于亚马逊的平台）标注的，因此问答种类相对比较宽泛自由，要训练这种数据应该需要更加可靠的模型，但是生成的问答也是最接近自然的。另外有趣的是大部分的问答都是由中文翻译过去的，也就是说如果我们能拿到原始数据集（中文），我们就可以构建中文VQA了
    2. 数据集地址：找了一圈没找到。
    3. 图片数量：120,360
    4. 问题数量: 250,560
    5. 答案数量: 250,560
 
* Visual Genome
    1. 数据集描述：应该是这次数据搜集中最大的数据集了。图片来源于 Visual Genome项目，包含大量的场景图。问题以七个W（即who, what, where, when, why, how, which)开头, 另外分free-form和region-based两类。该数据集的问答基数和多样性要比COCO-QA数据集大（在COCO中前1000个最频繁出现的答案几乎占总答案的80%， 但这个数据集之占64%）。
    2. 数据集[地址](https://visualgenome.org/api/v0/api_home.html)
    3. 图片数量: 
    4. 问题数量：
    5. 答案数量：

* Visual Madlibs：
    这个主要用来填空用，对我们意义不是很大

* 其他：
    如FVQA（主要用在专家系统中），Diagram，Shape等

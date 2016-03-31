
# Writer Identification
+ Image Processing project at Uni-Hamburg
+ `Author: Praveen Baburao Kulkarni`


| Download report | Download presentation |


## **About Project**

In this project, a framework for writer identification and different components needed to build it will be presented. The system uses five different kinds of features (Curvature, Directional, Edge Based Chain code and Tortuosity) extracted from the images. These features can then be used for classification using different classifier algorithms. We observed that changing the feature representation and then applying training classifiers improves the results considerably as it helps untangle and reveal the different explanatory factors of variation behind the data. In our work, we used two strategies for changing input feature representation namely dimensionality reduction and representation learning.

Two major contributions of this paper are:

+ Using Restricted Boltzmann Machines (RBMs) for representation learning in context of writer identification task and comparing it with Latent Dirichlet Allocation
+ Design of parallel framework for writer identification task

We report classification accuracy of around 90 percent by doing dimensionality reduction with Linear Support Vector Classifier followed by classification using Linear Discriminant Analysis (LDA). On the other hand, 89.7 percent accuracy was observed by doing  representation learning using RBMs followed by LDA for classification. The classification accuracy is equal to the state of art implementations which do not use RBMs. At the end of the paper, the parallel framework that was designed for writer identification and how all the components fit inside it will be presented.


## **Further Information and links**
+ [Environment and installing instructions](http://praveenneuron.github.io/writer_identification_doc/html/md_installation.html)
+ [Data Set description](http://praveenneuron.github.io/writer_identification_doc/html/md_about_dataset.html)
+ [Code and file Structure](http://praveenneuron.github.io/writer_identification_doc/html/md_code_structure.html)

**Browse source code**
+ [data handling](http://praveenneuron.github.io/writer_identification_doc/html/namespacesource_1_1data__handling.html)
+ [unit tests](http://praveenneuron.github.io/writer_identification_doc/html/unit__tests_8py.html)

**Report and results**
+ To be added soon ...

**Documentation is hosted on Github [(Click here)](http://praveenneuron.github.io/writer_identification_doc/html/index.html)**

**Download source code** [(Click here)](https://github.com/praveenneuron/WriterIdentification/archive/master.zip)
+ *Note: You need to have Github login and access to repository for downloading (It is private now)*
+ *But you can still browse the code inside doxygen*
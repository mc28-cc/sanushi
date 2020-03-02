from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import os
import pickle as pkl
import errno
import sys

import matplotlib.pyplot as plt
import matplotlib

import susi
from PIL import Image

from sklearn.datasets import make_blobs

###Feature extraction Functions

#function to extract VGG features from the image
#Input are filename and vgg model
#the image is resized to standard size for VGG16,
#feature are extracted and returned as np array

def featVGG16(vggModel, inputIMGFileName):
    if(os.path.isfile(inputIMGFileName) ):
        img = image.load_img(inputIMGFileName, target_size=(224, 224))#reshape image
        img_data = image.img_to_array(img)#get np array
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)#scale the input as the training set used for VGG

        vgg16_feature = vggModel.predict(img_data)#extract feature
        vgg16_feature_np = np.array(vgg16_feature).flatten()#reshape features


        return vgg16_feature_np
    else:
        return []

#wrapper function to save features (as pickle file) for each image in sourceFolder
#corresponding files are saved in destFolder
#it also accumulate a numpy matrix as panda frame with all the features corresponding to each image
#this will be reused later on in the sanushi program....

def saveFeatVGG16_folderContent(vggModel,sourceFolder,destFolder,imgType='.jpg'):
    if (os.path.isdir(sourceFolder)):
        #create dest folder....
        try:
            os.makedirs(destFolder,exist_ok=True)
        except OSError as exc:
            if(exc.errno != errno.EEXIST):
                print("Creation of the directory %s failed" % destFolder)
                return []
            pass
        else:
            print("Successfully created the directory %s " % destFolder)
            #take img list
            filenameList=[fn for fn in os.listdir(sourceFolder) if fn.endswith(imgType) ]
            #
            #for currentIMG in filenameList:
            accumulatedFeatures=[]
            for currentIMG in filenameList:
                absPath= sourceFolder + '/' + currentIMG
                currentFeat=featVGG16(vggModel,absPath)
                tmpName,tmpExt=os.path.splitext(os.path.basename(absPath))
                destFile=destFolder + '/' + tmpName + '.sav'

                fileFeatPKL = open(destFile, 'wb')
                pkl.dump(currentFeat, fileFeatPKL)
                fileFeatPKL.close()

                accumulatedFeatures.append(currentFeat)

            accumulatedFeaturesNP=np.array(accumulatedFeatures)
            accumulatedFeaturesPD=pd.DataFrame(accumulatedFeaturesNP)
            accumulatedFeaturesPD['filename']=filenameList

            destFile = destFolder + '/accumulatedFeatures.sav'
            #accumulatedFeaturesPD.to_csv(destFile,index=False)
            fileFeatPKL = open(destFile, 'wb')
            pkl.dump(accumulatedFeaturesPD, fileFeatPKL)
            fileFeatPKL.close()
            return accumulatedFeaturesPD
    else:
        print(sourceFolder + "does not exists!!!!!!")
        return []
#SOme plotting image functions
def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def make_array(imgListName):
    theArray=[np.asarray(Image.open(currentImage).convert('RGB').resize((224,224))) for currentImage in imgListName]
    return np.array(theArray)

if __name__ == "__main__":
    print("Sanushi test starting")

    #### some variables for paths and flags
    #Declare some paths here
    workingPath='.'
    #we are assuming that the images are stored in a subfolder of the workingPath, modify this if necessary
    rawImgFolder=workingPath  + '/sushi_or_sandwich/'
    rawImgFolderSushi = rawImgFolder + '/sushi/'
    rawImgFolderSandwich = rawImgFolder + '/sandwich/'

    featFolder=workingPath + '/datasetFeature/'#Path containing the extracted VGG features for the dataset
    featFolderSushi=featFolder + '/sushi/'
    featFolderSandwich=featFolder + '/sandwich/'
    modelFolder = workingPath + '/modelSOM/'
    resultsFolder = workingPath + '/resultsGraphs/'

    for currFolder in [featFolder,featFolderSushi,featFolderSandwich,modelFolder,resultsFolder]:
        try:
            os.makedirs(currFolder,exist_ok=True)

        except OSError as exc:
            if(exc.errno != errno.EEXIST):
                print("Creation of the directory %s failed" % currFolder)
                sys.exit(0)
            else:
                print("Successfully created the directory %s " % currFolder)

    # Declare some flags here
    FeatExON = True  # set it to True to extract for all the dataset images and save to disk.
                     # once calculated you can avoid this step if you re-run the algorithm

    SomFitON = True #set it to True to fit the SOM into the dataset. The model will be saved on the disk
                    # once calculated you can avoid this step if you re-run the algorithm

    if(FeatExON):
        #### FEATURE EXTRACTION
        print("Feature extraction")

        feVGGModel = VGG16(weights='imagenet', include_top=False)#take vgg model for feature extraction
        #sushi feat extraction
        allDataFeatSushi=saveFeatVGG16_folderContent(feVGGModel,rawImgFolderSushi,featFolderSushi)
        #sandwich feat extraction
        allDataFeatSandwich=saveFeatVGG16_folderContent(feVGGModel,rawImgFolderSandwich,featFolderSandwich)

    else:
        sushiFeatName=featFolderSushi + '/accumulatedFeatures.sav'
        fileFeatPKL = open(sushiFeatName, 'rb')
        allDataFeatSushi = pkl.load(fileFeatPKL)
        fileFeatPKL.close()
        sushiFeatName = featFolderSandwich + '/accumulatedFeatures.sav'
        fileFeatPKL = open(sushiFeatName, 'rb')
        allDataFeatSandwich = pkl.load(fileFeatPKL)
        fileFeatPKL.close()

    #a bit of data merging, adding labels as well....
    #we need to do this to be able to pass it to the
    #SOM clustering
    allDataFeatSushi['Class']=1#1 for sushi
    allDataFeatSandwich['Class']=2#2 for sandwich

                                                                                 
    #entireDataset=pd.concat([allDataFeatSushi.loc[0:10,],allDataFeatSandwich.loc[0:10,]],ignore_index=True)
    entireDataset=pd.concat([allDataFeatSushi,allDataFeatSandwich],ignore_index=True)
                                                                                 
    entireFeat=entireDataset.drop(['Class','filename'],axis=1).values#take the feature only
    featSushi=allDataFeatSushi.drop(['Class','filename'],axis=1).values
    featSandwich = allDataFeatSandwich.drop(['Class', 'filename'], axis=1).values

    #select number of som neurons as in
    numNeurons=15
    if(SomFitON):
        #Apply Som Clustering
        somClus = susi.SOMClustering(
            n_rows=numNeurons,
            n_columns=numNeurons,
            verbose = 1,
            n_jobs=1#change this to use the number of cores you desire in your local machine...
           )

        somClus.fit(entireFeat)
        #save the model on the disk....
        somModelName = modelFolder + '/fittedSomModel.sav'
        fileFeatPKL = open(somModelName, 'wb')
        pkl.dump(somClus, fileFeatPKL)
        fileFeatPKL.close()
        print("SOM fitted!")
    else:
        #just load the existing model file for the som...
        somModelName = modelFolder + '/fittedSomModel.sav'
        fileFeatPKL = open(somModelName, 'rb')
        somClus = pkl.load(fileFeatPKL)
        fileFeatPKL.close()
        print("Stored model loaded")


    #very simple code to test array of images for results ignore...
    #IGNORE IGNORE
    #imgListName=entireDataset['filename'].iloc[0:6].values
    #imgListNameComplete=[rawImgFolderSushi + currentName for currentName in imgListName.tolist()]
    #imgList = make_array(imgListNameComplete)
    #imgComposed=gallery(imgList, ncols=3)
    #plt.imshow(imgComposed)
    #plt.axis('off')
    #plt.show()



    ##GENERATE NODE DISTANCE MATRIX....
    figUMAP=plt.figure()

    u_matrix = somClus.get_u_matrix()
    plt.imshow(np.squeeze(u_matrix), cmap="Greys")
    plt.colorbar()
    #save umap
    figUMAP.tight_layout()
    figUMAP.savefig( resultsFolder+'/somUMAP.png', bbox_inches='tight')


    #Lets now use the obtained unsupervised som to understand how the samples, and the different classes
    #are distributed along the classes....

    #declaring here some empty matrices...
    totalNumberOfPointPerNode=np.zeros((numNeurons,numNeurons))
    totalNumberOfSushiPerNode=np.zeros((numNeurons,numNeurons))
    totalNumberOfSandwichPerNode=np.zeros((numNeurons,numNeurons))
    emptyNodes=np.zeros((numNeurons,numNeurons))
    dominantClassNode=np.zeros((numNeurons,numNeurons))
    pureShushiNode=np.zeros((numNeurons,numNeurons))
    pureSandwichNode=np.zeros((numNeurons,numNeurons))
    mixedNode=np.zeros((numNeurons,numNeurons))
    clusComposition = np.zeros((numNeurons, numNeurons))

    #now for each node in the SOM map
    #calculate the number of points belonging to it. Remember that SOM allows to have empty nodes!!!!!
    #this is a loop that may be optimized, but som map is not that big...it should not cause too much trouble
    bmu_listEntireDataset = somClus.get_bmus(entireFeat)#this function gives you the Best Matching Unit for each datapoint
    for currentBMU in bmu_listEntireDataset:
        xComponent=currentBMU[0]
        yComponent=currentBMU[1]
        totalNumberOfPointPerNode[xComponent,yComponent]=totalNumberOfPointPerNode[xComponent,yComponent]+1

    #plot and save number per sample graph
    figNumSample=plt.figure()
    plt.imshow(np.squeeze(totalNumberOfPointPerNode), cmap="gray")
    plt.xlabel('SOM COLUMNS')
    plt.ylabel('SOM ROWS')
    plt.colorbar()
    figNumSample.savefig(resultsFolder + '/somSamplePerNode.png', bbox_inches='tight')

    #now for each node in the SOM map
    #calculate dominantClass
    bmu_listSushi = somClus.get_bmus(featSushi)#this function gives you the Best Matching Unit for each datapoint
    for currentBMU in bmu_listSushi:
        xComponent=currentBMU[0]
        yComponent=currentBMU[1]
        totalNumberOfSushiPerNode[xComponent,yComponent]=totalNumberOfSushiPerNode[xComponent,yComponent]+1

    bmu_listSandwic = somClus.get_bmus(featSandwich)#this function gives you the Best Matching Unit for each datapoint
    for currentBMU in bmu_listSandwic:
        xComponent=currentBMU[0]
        yComponent=currentBMU[1]
        totalNumberOfSandwichPerNode[xComponent,yComponent]=totalNumberOfSandwichPerNode[xComponent,yComponent]+1


    #dominant class plotting...., we mark for graphic purposes with 100
    #the clusters where a higher number of sushi image is found,
    #viceversa we mark with 30 the one with a majority of sandwhich images.
    #FInally, we mark with 0 empty cluster and 60 the nodes with same number of element per class
    maskDominantSushi=totalNumberOfSushiPerNode>totalNumberOfSandwichPerNode
    maskDominantSanwich=totalNumberOfSushiPerNode<totalNumberOfSandwichPerNode
    maskEqual = totalNumberOfSushiPerNode == totalNumberOfSandwichPerNode
    maskZero= totalNumberOfPointPerNode ==0
    dominantClassNode[maskDominantSushi]=100
    dominantClassNode[maskEqual]=60
    dominantClassNode[maskDominantSanwich]=30
    dominantClassNode[maskZero]=0

    #print here figure where for each node we report the dominant class
    figDomClass = plt.figure()
    plt.imshow(np.squeeze(dominantClassNode), cmap="coolwarm")
    plt.xlabel('SOM COLUMNS')
    plt.ylabel('SOM ROWS')
    plt.title('Dominant Class Plot')
    figDomClass.savefig(resultsFolder + '/dominantClassPerNode.png', bbox_inches='tight')

    #now for each node in the SOM map
    #calculate pure clusters and mixed clusters. This is to understand how many mixed nodes are present, and how they
    #are distributed with respect to the pure-nodes containing only sushi or sandwich
    pureShushiNode=np.logical_and(totalNumberOfSushiPerNode>0,totalNumberOfSandwichPerNode==0)
    pureSandwichNode=np.logical_and(totalNumberOfSandwichPerNode>0,totalNumberOfSushiPerNode==0)
    mixedNode=np.logical_and(totalNumberOfSandwichPerNode>0,totalNumberOfSushiPerNode>0)

    #build here a matrix for representation purposes, pure sushi marked with 100, mixed node with 60, sandwich with 30
    # and as usual empty clusters with 0
    clusComposition[pureShushiNode]=100
    clusComposition[mixedNode]=60
    clusComposition[pureSandwichNode]=30
    clusComposition[maskZero]=0

    figCompositionClus = plt.figure()
    plt.imshow(np.squeeze(clusComposition), cmap="RdYlGn")
    plt.xlabel('SOM COLUMNS')
    plt.ylabel('SOM ROWS')
    plt.title('Cluster Composition Class Plot')
    figCompositionClus.savefig(resultsFolder + '/nodeComposition.png', bbox_inches='tight')

    #plot some image contained in the nodes example.....
    #select the largest node containing all sushi images....
    pureSushiCount=totalNumberOfSushiPerNode.copy()
    pureSushiCount[pureSandwichNode]=0
    pureSushiCount[mixedNode]=0
    maxIndex=np.argmax(pureSushiCount)
    tupleIndex=np.unravel_index(maxIndex, np.array(pureSushiCount).shape)
    print("Largest Sushi Node Coordinates in SOM map")
    print(tupleIndex)
    print("Total of sushi images on pure sushi nodes: " + str(np.sum(pureSushiCount)))
    listOfImagesIndex = somClus.get_datapoints_from_node(tupleIndex)
    listOfImagesALLSUSHI=entireDataset['filename'].iloc[listOfImagesIndex].values.tolist()
    print("List of images from the largest pure sushi node")
    print(listOfImagesALLSUSHI)

    #add the folder here plot the elements!!!!!!!!
    #Be aware arranging here image in the grid is done manually when setting ncols=7 in gallery
    imgListNameComplete=[rawImgFolderSushi + '/' + currentName for currentName in listOfImagesALLSUSHI]
    imgList = make_array(imgListNameComplete)
    imgComposed=gallery(imgList, ncols=7)
    figCompositionSUSHI=plt.figure()
    plt.imshow(imgComposed)
    plt.axis('off')
    figCompositionSUSHI.savefig(resultsFolder + '/pureSushiExample.png', bbox_inches='tight')

    #select the largest node containing all sandwich images....
    pureSanwichCount=totalNumberOfSandwichPerNode.copy()
    pureSanwichCount[pureShushiNode]=0
    pureSanwichCount[mixedNode]=0
    maxIndex=np.argmax(pureSanwichCount)
    tupleIndex=np.unravel_index(maxIndex, np.array(pureSanwichCount).shape)
    print("Largest Sandwich Node Coordinates in SOM map")
    print(tupleIndex)
    print("Total of sandwich images on pure sandwich nodes: " + str(np.sum(pureSanwichCount)))
    listOfImagesIndex = somClus.get_datapoints_from_node(tupleIndex)
    listOfImagesALLSANDWICH=entireDataset['filename'].iloc[listOfImagesIndex].values.tolist()
    print("List of images from the largest pure sandwich node")
    print(listOfImagesALLSANDWICH)

    #add the folder here plot the elements!!!!!!!!
    #Be aware arranging here image in the grid is done manually when setting ncols=7 in gallery
    imgListNameComplete=[rawImgFolderSandwich + '/' + currentName for currentName in listOfImagesALLSANDWICH]
    imgList = make_array(imgListNameComplete)
    imgComposed=gallery(imgList, ncols=4)
    figCompositionSAND=plt.figure()
    plt.imshow(imgComposed)
    plt.axis('off')
    figCompositionSAND.savefig(resultsFolder + '/pureSandExample.png', bbox_inches='tight')


    # select the first largest nodes containing a 50-50 population
    dominantClassNode[maskEqual] = 60

    balancedNodeCount=totalNumberOfPointPerNode.copy()
    balancedNodeCount[maskDominantSanwich]=0
    balancedNodeCount[maskDominantSushi] = 0
    maxIndex=np.argmax(balancedNodeCount)
    tupleIndex=np.unravel_index(maxIndex, np.array(balancedNodeCount).shape)
    print("Largest balanced-class Node Coordinates in SOM map for largest balanced ")
    print(tupleIndex)
    listOfImagesIndex = somClus.get_datapoints_from_node(tupleIndex)
    listOfImagesBalanced=entireDataset[['filename','Class']].iloc[listOfImagesIndex]
    print("List of images from the largest class-balanced node")
    print(listOfImagesBalanced)

    #here you have to remember how the dataset was structured just to pick images from right folder sushi/sandwich
    #use class column to complete this selection
    listOfImagesBalancedSushi=listOfImagesBalanced[listOfImagesBalanced['Class']==1]
    listOfImagesBalancedSandwich=listOfImagesBalanced[listOfImagesBalanced['Class']==2]

    tmpListNameCompleteSandwich=[rawImgFolderSandwich + '/' + currentName
                         for currentName in listOfImagesBalancedSandwich['filename'].values.tolist()]

    tmpListNameCompleteSushi=[rawImgFolderSushi + '/' + currentName
                         for currentName in listOfImagesBalancedSushi['filename'].values.tolist()]


    imgListNameComplete = tmpListNameCompleteSushi + tmpListNameCompleteSandwich
    imgList = make_array(imgListNameComplete)
    imgComposed = gallery(imgList, ncols=5)
    figCompositionSAND = plt.figure()
    plt.imshow(imgComposed)
    plt.axis('off')
    figCompositionSAND.savefig(resultsFolder + '/pureBalancedExample.png', bbox_inches='tight')

    #now examine the largest mixed node

    mixedNodeCount = totalNumberOfPointPerNode.copy()
    mixedNodeCount[pureShushiNode] = 0
    mixedNodeCount[pureSandwichNode] = 0
    maxIndex = np.argmax(mixedNodeCount)
    tupleIndex = np.unravel_index(maxIndex, np.array(mixedNodeCount).shape)
    print("Largest mixed-class Node Coordinates in SOM map for largest balanced ")
    print(tupleIndex)
    listOfImagesIndex = somClus.get_datapoints_from_node(tupleIndex)
    listOfImagesMixed = entireDataset[['filename', 'Class']].iloc[listOfImagesIndex]
    print("List of images from the largest mixed-balanced node")
    print(listOfImagesMixed)

    #here you have to remember how the dataset was structured just to pick images from right folder sushi/sandwich
    #use class column to complete this selection
    listOfImagesMixedSushi=listOfImagesMixed[listOfImagesMixed['Class']==1]
    listOfImagesMixedSandwich=listOfImagesMixed[listOfImagesMixed['Class']==2]

    tmpListNameCompleteSandwich=[rawImgFolderSandwich + '/' + currentName
                         for currentName in listOfImagesMixedSandwich['filename'].values.tolist()]

    tmpListNameCompleteSushi=[rawImgFolderSushi + '/' + currentName
                         for currentName in listOfImagesMixedSushi['filename'].values.tolist()]


    imgListNameComplete = tmpListNameCompleteSushi + tmpListNameCompleteSandwich
    #note here we have 23 images, we separate them as two different images of 15 and 8
    #to have two nice grids

    imgList = make_array(imgListNameComplete[0:15])
    imgComposed = gallery(imgList, ncols=5)
    figCompositionSAND = plt.figure()
    plt.imshow(imgComposed)
    plt.axis('off')
    figCompositionSAND.savefig(resultsFolder + '/mixedExamplePart1.png', bbox_inches='tight')

    imgList = make_array(imgListNameComplete[15:])
    imgComposed = gallery(imgList, ncols=4)
    figCompositionSAND = plt.figure()
    plt.imshow(imgComposed)
    plt.axis('off')
    figCompositionSAND.savefig(resultsFolder + '/mixedExamplePart2.png', bbox_inches='tight')

    #Finally try a different mixed node composition 75/25 to identify sanushi nodes
    dominantClassNode25=np.zeros((numNeurons,numNeurons))
    maskZero= totalNumberOfPointPerNode ==0
    maskDominantSushi = np.logical_and(totalNumberOfSushiPerNode > totalNumberOfSandwichPerNode,
                                       totalNumberOfSushiPerNode >= totalNumberOfPointPerNode*0.75)

    maskDominantSanwich = np.logical_and(totalNumberOfSushiPerNode < totalNumberOfSandwichPerNode,
                                         totalNumberOfSandwichPerNode >= totalNumberOfPointPerNode * 0.75)

    maskMixed = np.logical_and(totalNumberOfSandwichPerNode > totalNumberOfPointPerNode * 0.25,
                               totalNumberOfSushiPerNode > totalNumberOfPointPerNode * 0.25)
    dominantClassNode25[maskDominantSushi] = 100
    dominantClassNode25[maskMixed] = 60
    dominantClassNode25[maskDominantSanwich] = 30
    dominantClassNode25[maskZero] = 0

    # print here figure where for each node we report the dominant class
    figDomClass = plt.figure()
    plt.imshow(np.squeeze(dominantClassNode25), cmap="coolwarm")
    plt.xlabel('SOM COLUMNS')
    plt.ylabel('SOM ROWS')
    plt.title('Dominant Class Plot (75%-25% balance on mixed nodes)')

    figDomClass.savefig(resultsFolder + '/dominantClassPerNodePRC25.png', bbox_inches='tight')


    plt.show()
    print("Sanushi test Completed")
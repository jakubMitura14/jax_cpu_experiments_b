import functools
import multiprocessing as mp
import os
from functools import partial
from zipfile import BadZipFile, ZipFile
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sys, time, os
import numpy as np
import shutil
import glob
import pydicom
from pydicom import dcmread
import pydicom as pyd
from pydicom.pixel_data_handlers.util import convert_color_space
import os
import subprocess
import tempfile
import uuid
from pydicom.uid import generate_uid
from pydicom.uid import UID
from medpy.io import load

#main part adapted from https://simpleitk.readthedocs.io/en/next/Examples/DicomSeriesFromArray/Documentation.html
#location in dicom calculations https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html#sect_C.7.6.2.1.1

targetDir= '/workspaces/jax_cpu_experiments_b/explore/picai_unpacked'
sample_dicom_fp="/workspaces/jax_cpu_experiments_b/explore/xnat/003/003/003_MR_1/scans/3-ep2d_diff_b_50_400_800_1200_TRACEW/resources/DICOM/files/1.3.12.2.1107.5.8.15.100960.30000022021714130657000000014-3-94-ju2ghp.dcm"
dirDict={}
for subdir, dirs, files in os.walk(targetDir):
    for subdirin, dirsin, filesin in os.walk(subdir):
        lenn= len(filesin)
        if(lenn>0):
            try:
                dirDict[subdirin.split("/")[5]]=filesin
            except:
                pass
            # print(f"subdir {subdir}")
            #print(subdir.split("/"))
            # print(subdirin.split("/"))
            #dirDict[subdir]=filesin







def writeSlices(series_tag_values, new_img, i,writer,study_dir_path,patient_id):
    image_slice = new_img[:,:,i]
    image_slice.SetMetaData("0008|0050", str(i).strip()) # Accession number

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
    image_slice.SetMetaData("0020|0013", str(i).strip()) # Instance Number
    # image_slice.SetMetaData("0020|0012", str(i)) # Acquisition Number   
    # image_slice.SetMetaData("0020|0013", str(i)) # Instance Number  

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "MR")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)


    # image_slice.SetMetaData("0020,0010", study_id) # study_id


    # study instance uid 20 0000d

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    out_path=os.path.join(study_dir_path,str(i)+'.dcm')
    writer.SetFileName(out_path)
    writer.Execute(image_slice)
    return out_path


def get_modality_tags(tag_name):
    """
    manually set tags diffrent for each modality
    return Series Number ;Window Center  ; Window Width 
    """
    if(tag_name=='t2w'):
        return (0,424,808,365 ,"Tra","ORIGINAL, PRIMARY, M, NORM, DIS2D, SH, FIL	CS	36","axial")
    if(tag_name=='sag'):
        return (1,424,808,365,"Sag" ,"ORIGINAL, PRIMARY, M, NORM, DIS2D, FS, FIL	CS	36","sagittal")
    if(tag_name=='adc'):
        return (3,814,1716,2205,"Tra" ,"DERIVED, PRIMARY, DIFFUSION, ADC, DIS2D	CS	36","axial")
    if(tag_name=='hbv'):
        return (4,31,81,2205 ,"Tra","DERIVED, PRIMARY, DIFFUSION, TRACEW, DIS2D	CS	38","axial")
    if(tag_name=='cor'):
        return (5,424,808,365,"Cor" ,"[0008,0008]	ImageType	[7] ORIGINAL, PRIMARY, M, NORM, DIS2D, FS, FIL	CS	36","coronal")
        

# ['1', '0', '0', '0', '0', '-1'] you are dealing with Coronal plane view
# ['0', '1', '0', '0', '0', '-1'] you are dealing with Sagittal plane view
# ['1', '0', '0', '0', '1', '0'] you are dealing with Axial plane view

        
# def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
#     original_spacing = itk_image.GetSpacing()
#     original_size = itk_image.GetSize()

#     out_size = [
#         int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
#         int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
#         int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
#     ]

#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(out_spacing)
#     resample.SetSize(out_size)
#     resample.SetOutputDirection(itk_image.GetDirection())
#     resample.SetOutputOrigin(itk_image.GetOrigin())
#     resample.SetTransform(sitk.Transform())
#     resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

#     if is_label:
#         resample.SetInterpolator(sitk.sitkNearestNeighbor)
#     else:
#         resample.SetInterpolator(sitk.sitkBSpline)

#     return resample.Execute(itk_image) 



def convert_mha_file(image_path,patient_dir_path,patient_id,sample_dicom_fp,Frame_of_Reference_UID):
    """Takes path to an mha file converts and saves as series of dicom files.
    filepath: str path to an mha file.
    output_dir:str root directory where images should be stored.
    plane: Acquisition plane  of the original image.
    sample_dicom_fp: Path to dicom image to use as reference
    window_center: Dicom Parameter  used for windowing
    window_width: Dicom Parameter used for windowing
    """
    output_dir=patient_dir_path
    filepath=image_path
    input_ext = '.mha'
    voxel_arr,header = load(filepath)

    modality_str=filepath.split('_')[-1].replace('.mha','')
    Series_Number,Window_Center,Window_Width,Pixel_Bandwidth,Plane,Image_type,plane =get_modality_tags(modality_str)
    window_center=Window_Center
    window_width=Window_Width

    pixdim = header.get_voxel_spacing()

    # Image coordinates -> World coordinates
    if plane == "axial":
        slice_axis = 2
        plane_axes = [0, 1]
    elif plane == "coronal":
        slice_axis = 1
        plane_axes = [0, 2]
    elif plane == "sagittal":
        slice_axis = 0
        plane_axes = [1, 2]
    thickness = pixdim[slice_axis]
    spacing = [pixdim[plane_axes[1]], pixdim[plane_axes[0]]]

    # generate DICOM UIDs (StudyInstanceUID and SeriesInstanceUID)
    study_uid = pyd.uid.generate_uid(prefix=None)
    series_uid = pyd.uid.generate_uid(prefix=None)
    # randomized patient ID

    # patient_id = str(UID.uuid.uuid4())
    # patient_name = patient_id

    scale_slope = "1"
    scale_intercept = "0"
    # create base directory
    base_dir =  os.path.join( output_dir,modality_str)
    os.makedirs(base_dir,exist_ok=True)
    for slice_index in range(voxel_arr.shape[-1]):
        # generate SOPInstanceUID
        instance_uid = pyd.uid.generate_uid(prefix=None)
        if(slice_index==0 and modality_str=='t2w'):
            instance_uid=Frame_of_Reference_UID
        loc = slice_index * thickness

        ds = pyd.dcmread(sample_dicom_fp)

        # delete tags
        del ds[0x00200052]  # Frame of Reference UID
        del ds[0x00201040]  # Position Reference Indicator
        del ds[0x00080008] 

        # slice and set PixelData tag
        axes = [slice(None)] * 3
        axes[slice_axis] = slice_index
        arr = voxel_arr[:,:,slice_index].T.astype(np.int16)
        ds[0x7fe00010].value = arr.tobytes()

        # modify tags
        # using code from original nifti2dcm
        # - UIDs are created by pydicom.uid.generate_uid at each level above
        # - image position is calculated by combination of slice index and slice thickness
        # - slice location is set to the value of image position along z-axis
        # - Rows/Columns determined by array shape
        # - we set slope/intercept to 1/0 since we're directly converting from PNG pixel values
        ds[0x00080018].value = instance_uid  # SOPInstanceUID
        ds[0x00100010].value = patient_id
        ds[0x00100020].value = patient_id
        ds[0x0020000d].value = study_uid  # StudyInstanceUID
        ds[0x0020000e].value = series_uid  # SeriesInstanceUID
        ds[0x0008103e].value = modality_str  # Series Description
        ds[0x00200011].value = Series_Number  # Series Number
        ds[0x00200012].value = str(slice_index + 1)  # Acquisition Number
        ds[0x00200013].value = str(slice_index + 1)  # Instance Number
        ds[0x00201041].value = str(loc)  # Slice Location
        ds[0x00280010].value = arr.shape[0]  # Rows
        ds[0x00280011].value = arr.shape[1]  # Columns
        ds[0x00280030].value = spacing  # Pixel Spacing
        ds[0x00281050].value = str(window_center)  # Window Center
        ds[0x00281051].value = str(window_width)  # Window Width
        
        ds.add_new([0x0028, 0x1052], 'DS', scale_intercept)
        # ds[0x00281052].value = str(scale_intercept)  # Rescale Intercept
        
        ds.add_new([0x0028, 0x1053], 'DS', scale_slope)
        # ds[0x00281053].value = str(scale_slope)  # Rescale Slope
        ds.add_new([0x0008, 0x0008], 'LO', str(Image_type))

        ds.add_new([0x0020, 0x0052], 'LO', Frame_of_Reference_UID)

        # elem.value = str(Frame_of_Reference_UID)  # Frame_of_Reference_UID

        ds.Modality = "MR"

        # Image Position (Patient)
        # Image Orientation (Patient)
        if plane == "axial":
            ds[0x00200032].value = ["0", "0", str(loc)]
            ds[0x00200037].value = ["1", "0", "0", "0", "1", "0"]
        elif plane == "coronal":
            ds[0x00200032].value = ["0", str(loc), "0"]
            ds[0x00200037].value = ["1", "0", "0", "0", "0", "1"]
        elif plane == "sagittal":
            ds[0x00200032].value = [str(loc), "0", "0"]
            ds[0x00200037].value = ["0", "1", "0", "0", "0", "1"]



        # add new tags
        # see tag info e.g., from https://dicom.innolitics.com/ciods/nm-image/nm-reconstruction/00180050
        # Slice Thickness
        ds[0x00180050] = pyd.dataelem.DataElement(0x00180050, "DS", str(thickness))
        ds.SeriesDescription = f"MR {plane}"

        dicom_fp =  os.path.join(base_dir,
        "{:03}.dcm".format(slice_index + 1),
        )
        dcm_base,_ = os.path.split(dicom_fp)
        os.makedirs(dcm_base,exist_ok=True)
        pyd.dcmwrite(dicom_fp,ds)



def save_file_as_dicom(image_path,patient_dir_path,patient_id,t2w_orig,t2w_dir):
    modality_str=image_path.split('_')[-1].replace('.mha','')


    # print(f"image_path {image_path}   \n modality_str {modality_str}")
    new_img=sitk.ReadImage(image_path)
    study_dir_path=os.path.join(patient_dir_path,modality_str)
    os.makedirs(study_dir_path)

    arr= sitk.GetArrayFromImage(new_img)
    # if(not (modality_str=='sag' or modality_str=='cor')):
        # new_img.SetOrigin(t2w_orig) 
        # new_img.SetDirection(t2w_dir) 
        # new_img.SetMetaData("Offset",offset_t2w) 

    print(f"sssssssssss {modality_str} {arr.shape}")
    max_val= np.max(arr.flatten())
    min_val= np.min(arr.flatten())
    # Create a new series from a numpy array
    # new_arr = np.random.uniform(-10, 10, size = (3,4,5)).astype(np.int16)
    # new_img = sitk.GetImageFromArray(new_arr)
    # new_img.SetSpacing([2.5,3.5,4.5])

    # ustaw tagi ze sa saggital coronal ...
    # z jakiegos powodu saggital gubi koordynacje z transwersami

    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate opration and requires knowlege of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
    #            If it is critical for your work to generate valid DICOM files,
    #            It is recommended to use David Clunie's Dicom3tools to validate the files 
    #                           (http://www.dclunie.com/dicom3tools.html).

    writer = sitk.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:

    series=f"{patient_id}_{modality_str}"

    Series_Number,Window_Center,Window_Width,Pixel_Bandwidth,Plane,Image_type =get_modality_tags(modality_str)

    print(f"series {series}")
    direction = new_img.GetDirection()
    series_tag_values = [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED"), # Image Type


                    ("0020|000D", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+patient_id), # Series Instance UID
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modality_str+patient_id), # Series Instance UID
                    # ("0020|000e", patient_id), # Series Instance UID
                    # ("0020|000e", f"{patient_id}_{modality_str}"), # Series Instance UID
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                    ("0008|103e", f"{modality_str}"),# Series Description
                    # ("0008|103e", f"{patient_id}_{modality_str}"),# Series Description
                    ("0008|0060", modality_str),# modality description
                    
                    ("0020|0011", f"{Series_Number}"), 
                    ("0028|1050", f"{Window_Center}"), 
                    ("0028|1051", f"{Window_Width}"), 
                    ("0018|0095", f"{Pixel_Bandwidth}"),
                    ("0051|100E", f"{Plane}") ,
                    ("0008|0008", f"{Image_type}") ,
                    # ("0028|0106", f"{min_val}"), 
                    # ("0028|0107", f"{max_val}"), 
                    ] 
    # if(not (modality_str=='sag' or modality_str=='cor')):
    #     series_tag_values.append(("0018|0088", f"{new_img.GetSpacing()[2]}") )
    #     print(f" spacing {new_img.GetSpacing()}")

    # if(modality_str=='sag'):
    #     series_tag_values.append(("0018|0088", f"{new_img.GetSpacing()[1]}") )
    # if(modality_str=='cor'):
    #     series_tag_values.append(("0018|0088", f"{new_img.GetSpacing()[0]}") )        

# (0020|1041)      	Slice Location                     	DS	1	16        	9.5766525268555
    # Write slices to output directory
    out_paths=list(map(lambda i: writeSlices(series_tag_values, new_img, i,writer,study_dir_path,patient_id), range(new_img.GetDepth())))
    # Re-read the series
    # Read the original series. First obtain the series file names using the
    # image series reader.
    data_directory = study_dir_path
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)

    #set slice metadata after saving modifying just this one tag

    

    for file_name in out_paths:
        ds = dcmread(file_name)
        elem = ds[0x0018, 0x0088]
        # elem.value ="5"
        if(not (modality_str=='sag' or modality_str=='cor')):
            elem.value = f"{new_img.GetSpacing()[2]}"
        if(modality_str=='sag'):
            elem.value = f"{new_img.GetSpacing()[1]}"
        if(modality_str=='cor'):
            elem.value = f"{new_img.GetSpacing()[0]}"          
        ds.save_as(file_name)

        ds = dcmread(file_name)
        ds.add_new([0x0018, 0x0050], 'DS', "2.0")
        elem = ds[0x0018, 0x0050]
        # elem.value ="5"
        if(not (modality_str=='sag' or modality_str=='cor')):
            elem.value = f"{new_img.GetSpacing()[2]*0.8}"
        if(modality_str=='sag'):
            elem.value = f"{new_img.GetSpacing()[1]*0.8}"
        if(modality_str=='cor'):
            elem.value = f"{new_img.GetSpacing()[0]*0.8}"          
        ds.save_as(file_name)










    if not series_IDs:
        print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
        sys.exit(1)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    series_reader.LoadPrivateTagsOn()
    # for k in series_reader.GetMetaDataKeys(slice=1):
    #     v = series_reader.GetMetaData(k)
    #     print(f'({k}) = = "{v}"')
            
    image3D = series_reader.Execute()
    print(image3D.GetSpacing(),'vs',new_img.GetSpacing())
    # except Exception as e: print(f"eeeeeeeeeeee {e}")


def save_patient_files_as_dicom(tupl,orig_targetDir, new_main_target_dir):
    master_id=tupl[0]
    file_names=tupl[1]

    patient_dir_path=os.path.join(new_main_target_dir,master_id)
    os.makedirs(patient_dir_path,exist_ok=True)

    file_names=list(map(lambda file_name:os.path.join(orig_targetDir,master_id,file_name),file_names))
    t2w_file= list(filter(lambda pathh: 't2w' in pathh,file_names))[0]
    t2w_img=sitk.ReadImage(t2w_file)
    t2w_orig= t2w_img.GetOrigin()
    t2w_dir= t2w_img.GetDirection()
    Frame_of_Reference_UID= pyd.uid.generate_uid(prefix=None)
    
    # print(f"GetMetaDataKeys  {t2w_img.GetMetaDataKeys()}")
    # offset_t2w=t2w_img.GetMetaData("Offset") 

    # list(map(lambda image_path:save_file_as_dicom(image_path,patient_dir_path,master_id,t2w_orig,t2w_dir), file_names ))
    list(map(lambda image_path:convert_mha_file(image_path,patient_dir_path,master_id,sample_dicom_fp,Frame_of_Reference_UID), file_names ))
    


new_main_target_dir='/workspaces/jax_cpu_experiments_b/explore/picai_dicom'
shutil.rmtree(new_main_target_dir)
os.makedirs(new_main_target_dir)
dir_list = [(k, v) for k, v in dirDict.items()]
# print(dir_list)


list(map(lambda tupl : save_patient_files_as_dicom(tupl,targetDir, new_main_target_dir),dir_list[0:2]))


#print based on https://simpleitk.readthedocs.io/en/master/link_DicomImagePrintTags_docs.html

reader = sitk.ImageFileReader()

# reader.SetFileName("/workspaces/jax_cpu_experiments_b/explore/picai_dicom/10163/adc/1.dcm")
# reader.LoadPrivateTagsOn()

# reader.ReadImageInformation()

# for k in reader.GetMetaDataKeys():
#     v = reader.GetMetaData(k)
#     print(f'({k}) = = "{v}"')


# print("**********************************************************************************")


# reader.SetFileName("/workspaces/jax_cpu_experiments_b/explore/picai_dicom/10163/adc/2.dcm")
# reader.LoadPrivateTagsOn()

# reader.ReadImageInformation()

# for k in reader.GetMetaDataKeys():
#     v = reader.GetMetaData(k)
#     print(f'({k}) = = "{v}"')

# print("**********************************************************************************")


# def print_slice_loc(pathh):
#     reader.SetFileName(pathh)
#     reader.LoadPrivateTagsOn()
#     reader.ReadImageInformation()
#     print(f"(0020|1041) slice loc {reader.GetMetaData('0020|1041')}")
#     print(f"(0008,0050) Accession  {reader.GetMetaData('0008|0050')}")


# path_a="/workspaces/jax_cpu_experiments_b/explore/xnat/003/003/003_MR_1/scans/4-ep2d_diff_b_50_400_800_1200_ADC/resources/DICOM/files/1.3.12.2.1107.5.8.15.100960.30000022021714130657000000014-4-1-ju2h2z.dcm"
# path_b="/workspaces/jax_cpu_experiments_b/explore/xnat/003/003/003_MR_1/scans/4-ep2d_diff_b_50_400_800_1200_ADC/resources/DICOM/files/1.3.12.2.1107.5.8.15.100960.30000022021714130657000000014-4-2-ju2h31.dcm"
# path_c="/workspaces/jax_cpu_experiments_b/explore/xnat/003/003/003_MR_1/scans/4-ep2d_diff_b_50_400_800_1200_ADC/resources/DICOM/files/1.3.12.2.1107.5.8.15.100960.30000022021714130657000000014-4-3-ju2h32.dcm"

# print_slice_loc(path_a)
# print_slice_loc(path_b)
# print_slice_loc(path_c)

# for k in reader.GetMetaDataKeys():

#     v = reader.GetMetaData(k)
#     print(f'({k}) = = "{v}"')

# reader.SetFileName("/workspaces/jax_cpu_experiments_b/explore/picai_unpacked/10163/10000_1000000_adc.mha")
# reader.LoadPrivateTagsOn()

# reader.ReadImageInformation()

# for k in reader.GetMetaDataKeys():
#     v = reader.GetMetaData(k)
#     print(f'({k}) = = "{v}"')

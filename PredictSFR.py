#exec(open("./PredictSFR.py").read())
#PredictSFR([24])

def JytoLs(wave,flux,D):
	return flux * (np.pi * D**2) * 2.96E-11 / wave 



def PredictSFR(wave):

	# Load libraries
	import pandas
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn import model_selection
	from sklearn import linear_model 
	from sklearn.metrics import r2_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model import Ridge
	from sklearn.model_selection import cross_val_score
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.pipeline import make_pipeline
	from sklearn.pipeline import Pipeline


	# download the dataset:
	fluxes=['name',				#0
		'ra',					#1
		'dec',					#2
		'semimaj_arcsec',		#3
		'axial_ratio',			#4
		'pos_angle',			#5
		'global_flag',			#6
		'GALEX_FUV',			#7
		'GALEX_FUV_err',		#8
		'GALEX_FUV_flag',		#9
		'GALEX_NUV',			#10
		'GALEX_NUV_err',		#11
		'GALEX_NUV_flag',		#12
		'SDSS_u',				#13
		'SDSS_u_err',			#14
		'SDSS_u_flag',			#15
		'SDSS_g',				#16
		'SDSS_g_err',			#17
		'SDSS_g_flag',			#18
		'SDSS_r',				#19
		'SDSS_r_err',			#20
		'SDSS_r_flag',			#21
		'SDSS_i',				#22
		'SDSS_i_err',			#23
		'SDSS_i_flag',			#24
		'SDSS_z',				#25
		'SDSS_z_err',			#26		
		'SDSS_z_flag',			#27
		'2MASS_J',				#28
		'2MASS_J_err',			#29
		'2MASS_J_flag',			#30
		'2MASS_H',				#31
		'2MASS_H_err',			#32
		'2MASS_H_flag',			#33
		'2MASS_Ks',				#34
		'2MASS_Ks_err',			#35
		'2MASS_Ks_flag',		#36
		'WISE_3.4',				#37
		'WISE_3.4_err',			#38
		'WISE_3.4_flag',		#39
		'WISE_4.6',				#40
		'WISE_4.6_err',			#41
		'WISE_4.6_flag',		#42
		'WISE_12',				#43
		'WISE_12_err',			#44
		'WISE_12_flag',			#45
		'WISE_22',				#46
		'WISE_22_err',			#47
		'WISE_22_flag',			#48
		'Spitzer_3.6',			#49
		'Spitzer_3.6_err',		#50
		'Spitzer_3.6_flag',		#51
		'Spitzer_4.5',			#52
		'Spitzer_4.5_err',		#53
		'Spitzer_4.5_flag',		#54
		'Spitzer_5.8',			#55
		'Spitzer_5.8_err',		#56
		'Spitzer_5.8_flag',		#57
		'Spitzer_8.0',			#58
		'Spitzer_8.0_err',		#59
		'Spitzer_8.0_flag',		#60
		'Spitzer_24',			#61
		'Spitzer_24_err',		#62
		'Spitzer_24_flag',		#63
		'Spitzer_70',			#64
		'Spitzer_70_err',		#65
		'Spitzer_70_flag',		#66
		'Spitzer_160',			#67
		'Spitzer_160_err',		#68
		'Spitzer_160_flag',		#69
		'PACS_70',				#70
		'PACS_70_err',			#71
		'PACS_70_flag',			#72
		'PACS_100',				#73
		'PACS_100_err',			#74
		'PACS_100_flag',		#75
		'PACS_160',				#76
		'PACS_160_err',			#77
		'PACS_160_flag',		#78
		'SPIRE_250',			#79
		'SPIRE_250_err',		#80
		'SPIRE_250_flag',		#81
		'SPIRE_350',			#82
		'SPIRE_350_err',		#83
		'SPIRE_350_flag',		#84
		'SPIRE_500',			#85
		'SPIRE_500_err',		#86
		'SPIRE_500_flag']		#87

	# download the dataset:
	characteristics=['objname',
		'ra2000',
		'de2000',
		't',
		't_err',
		'type',
		'logd25',
		'd25',
		'bt',
		'incl',
		'ned_v_helio',
		'ned_v_corr',
		'ned_dist_z_corr',
		'ned_dist_0',
		'hyperleda_v_helio',
		'hyperleda_dist_z_helio',
		'hyperleda_dist_0',
		'dist_best']


	# Treat the wave argument
	usecols=[]
	usecols.append(0)
	wave.sort()
	for u in wave:
		if (u == 24): usecols.append(61)
		elif (u == 70): usecols.append(70)
		elif (u == 100): usecols.append(73)
		elif (u == 160): usecols.append(76)
		elif (u == 250): usecols.append(79)
		elif (u == 350): usecols.append(82)
		elif (u == 500): usecols.append(85)
		else: 
			print('Not a possible input wavelength; please choose 24, 70, 100, 160, 250')
			return None

	dataset = pandas.read_csv('DustPedia_Aperture_Photometry_2.2.csv', names=fluxes, dtype=None, skiprows=1, usecols = usecols)
	
	
	# Save the distances
	dataset2 = pandas.read_csv('DustPedia_HyperLEDA_Herschel.csv', names=characteristics, dtype=None, skiprows=1, usecols = [17])
	
	#dataset_noNan = dataset.dropna(axis=0)
	array = dataset.values
	array2 = dataset2.values*1e6

	result = []
	# Treat the third argument
	for i in range(len(array[:,0])):
		if (wave == [24]):  result = 10**(0.954*np.log10(JytoLs(24e-6,array[i,1],array2[i])) + 1.336)
		if (wave == [70]):  result = 10**(0.973*np.log10(JytoLs(70e-6,array[i,1],array2[i])) + 0.567)
		if (wave == [100]): result = 10**(1.000*np.log10(JytoLs(100e-6,array[i,1],array2[i])) + 0.256)
		if (wave == [160]): result = 10**(1.024*np.log10(JytoLs(160e-6,array[i,1],array2[i])) + 0.176)
		if (wave == [250]): result = 10**(1.060*np.log10(JytoLs(250e-6,array[i,1],array2[i])) + 0.451)		
# 	
		if (wave == [24,70]):   result =  3.98*JytoLs(24e-6,array[i,1],array2[i]) + 1.553*JytoLs(70,array[i,2],array2[i])	
		if (wave == [24,100]): 	result = 2.453*JytoLs(24e-6,array[i,1],array2[i]) + 1.407*JytoLs(100,array[i,2],array2[i])
		if (wave == [24,160]): 	result = 3.901*JytoLs(24e-6,array[i,1],array2[i]) + 1.365*JytoLs(160,array[i,2],array2[i])	
		if (wave == [24,250]): 	result = 5.288*JytoLs(24e-6,array[i,1],array2[i]) + 3.15 *JytoLs(250,array[i,2],array2[i])	
		if (wave == [70,100]): 	result = 0.463*JytoLs(70e-6,array[i,1],array2[i]) + 1.442*JytoLs(100,rray[:,2],array2[i])	
		if (wave == [70,160]): 	result = 1.01 *JytoLs(70e-6,array[i,1],array2[i]) + 1.218*JytoLs(160,array[i,2],array2[i])		
		if (wave == [70,250]): 	result = 1.325*JytoLs(70e-6,array[i,1],array2[i]) + 2.717*JytoLs(250,array[i,2],array2[i])	
		if (wave == [100,160]): result = 1.238*JytoLs(100e-6,array[i,1],array2[i]) + 0.62 *JytoLs(160,array[i,2],array2[i])		
		if (wave == [100,250]): result = 1.403*JytoLs(100e-6,array[i,1],array2[i]) + 1.242*JytoLs(250,array[i,2],array2[i])		
		if (wave == [160,250]): result = 2.37 *JytoLs(160e-6,array[i,1],array2[i]) - 1.029*JytoLs(250,array[i,2],array2[i])	
# 	
		if (wave == [24,70,100]):   result =  2.192*JytoLs(24e-6,array[i,1],array2[i]) + 0.187*JytoLs(70,array[i,2],array2[i])  + 1.314*JytoLs(100,array[i,3],array2[i])	
		if (wave == [24,70,160]):   result =  2.133*JytoLs(24e-6,array[i,1],array2[i]) + 0.681*JytoLs(70,array[i,2],array2[i])  + 1.125*JytoLs(160,array[i,3],array2[i])	
		if (wave == [24,70,250]):   result =  2.333*JytoLs(24e-6,array[i,1],array2[i]) + 0.938*JytoLs(70,array[i,2],array2[i])  + 2.49 *JytoLs(250,array[i,3],array2[i])	
		if (wave == [24,100,160]):  result =  2.739*JytoLs(24e-6,array[i,1],array2[i]) + 0.732*JytoLs(100,array[i,2],array2[i]) + 0.736*JytoLs(160,array[i,3],array2[i])
		if (wave == [24,100,250]):  result =  2.594*JytoLs(24e-6,array[i,1],array2[i]) + 0.99 *JytoLs(100,array[i,2],array2[i]) + 1.334*JytoLs(250,array[i,3],array2[i])
		if (wave == [24,160,250]):  result =  3.868*JytoLs(24e-6,array[i,1],array2[i]) + 1.458*JytoLs(160,array[i,2],array2[i]) - 0.252*JytoLs(250,array[i,3],array2[i])
		if (wave == [70,100,160]):  result =  0.808*JytoLs(70e-6,array[i,1],array2[i]) + 0.367*JytoLs(100,array[i,2],array2[i]) + 0.968*JytoLs(160,array[i,3],array2[i])
		if (wave == [70,100,250]):  result =  0.705*JytoLs(70e-6,array[i,1],array2[i]) + 0.784*JytoLs(100,array[i,2],array2[i]) + 1.639*JytoLs(250,array[i,3],array2[i])
		if (wave == [70,160,250]):  result =  1.032*JytoLs(70e-6,array[i,1],array2[i]) + 1.051*JytoLs(160,array[i,2],array2[i]) + 0.423*JytoLs(250,array[i,3],array2[i])
		if (wave == [100,160,250]): result =  1.379*JytoLs(100e-6,array[i,1],array2[i])+ 0.058*JytoLs(160,array[i,2],array2[i]) + 1.15 *JytoLs(250,array[i,3],array2[i])
# 	
		if (wave == [24,70,100,160]):  result =  2.064*JytoLs(24e-6,array[i,1],array2[i])  + 0.539*JytoLs(70e-6,array[i,2],array2[i]) + 0.277*JytoLs(100e-6,array[i,3],array2[i]) + 0.938*JytoLs(160e-6,array[i,4],array2[i])	
		if (wave == [24,70,100,250]):  result =  1.999*JytoLs(24e-6,array[i,1],array2[i])  + 0.443*JytoLs(70e-6,array[i,2],array2[i]) + 0.696*JytoLs(100e-6,array[i,3],array2[i]) + 1.563*JytoLs(250e-6,array[i,4],array2[i])	
		if (wave == [24,70,160,250]):  result =  2.127*JytoLs(24e-6,array[i,1],array2[i])  + 0.702*JytoLs(70e-6,array[i,2],array2[i]) + 0.974*JytoLs(160e-6,array[i,3],array2[i]) + 0.382*JytoLs(250e-6,array[i,4],array2[i])
		if (wave == [24,100,160,250]): result =  2.667*JytoLs(24e-6,array[i,1],array2[i])  + 0.848*JytoLs(100e-6,array[i,2],array2[i]) + 0.319*JytoLs(160e-6,array[i,3],array2[i]) + 0.847*JytoLs(250e-6,array[i,4],array2[i])
		if (wave == [70,100,160,250]): result =  0.783*JytoLs(70e-6,array[i,1],array2[i])  + 0.497*JytoLs(100e-6,array[i,2],array2[i]) + 0.54 *JytoLs(160e-6,array[i,3],array2[i]) + 0.852*JytoLs(250e-6,array[i,4],array2[i])
#	
		if (wave == [24,70,100,160,250]): 	result = 2.023*JytoLs(24e-6,array[i,1],array2[i]) + 0.523*JytoLs(70e-6,array[i,2],array2[i]) + 0.39*JytoLs(100e-6,array[i,3],array2[i]) + 0.577*JytoLs(160e-6,array[i,4],array2[i]) + 0.721*JytoLs(250e-6,array[i,5],array2[i])
# 
		print(array[i,0], 'SFR = ', 1.71E-10 * result[0])
















#PredictDustpedia([70,100,160,250,350],[2.7,2.8,5.3,2.8,1.2],500)
#PredictDustpedia([100,250],[2.8,2.8],350)

def PredictDustpedia(wave, flux, wavereq):

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
	names=['name',				#0
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


	# Treat the second argument
	usecols=[]
	for u in wave:
		if (u == 3.6): usecols.append(49)
		elif (u == 4.5): usecols.append(52)
		elif (u == 5.8): usecols.append(55)
		elif (u == 8): usecols.append(58)
		elif (u == 24): usecols.append(61)
		elif (u == 70): usecols.append(70)
		elif (u == 100): usecols.append(73)
		elif (u == 160): usecols.append(76)
		elif (u == 250): usecols.append(79)
		elif (u == 350): usecols.append(82)
		elif (u == 500): usecols.append(85)
		else: 
			print('Not a possible input wavelength; please choose 3.6, 4.5, 5.8, 8, 24, 70, 100, 160, 250, 350 or 500')
			return None
				
	# Treat the third argument
	if (wavereq == 3.6): usecols.append(49)
	elif (wavereq == 4.5): usecols.append(52)
	elif (wavereq == 5.8): usecols.append(55)
	elif (wavereq == 8 or wavereq == 8.0): usecols.append(58)
	elif (wavereq == 24 or wavereq == 24.0): usecols.append(61)
	elif (wavereq == 70 or wavereq == 70.0): usecols.append(70)	
	elif (wavereq == 100 or wavereq == 100.0): usecols.append(73)
	elif (wavereq == 160 or wavereq == 160.0): usecols.append(76)
	elif (wavereq == 250 or wavereq == 250.0): usecols.append(79)
	elif (wavereq == 350 or wavereq == 350.0): usecols.append(82)
	elif (wavereq == 500 or wavereq == 500.0): usecols.append(85)
	else: 
		print('Not a possible output wavelength; please choose 3.6, 4.5, 5.8, 8, 24, 70, 100, 160, 250, 350 or 500')
		return None
		
	print(usecols)

	# Treat the first argument
	wave = np.array(wave)*1E-6
	dataset = pandas.read_csv('DustPedia_Aperture_Photometry_2.2.csv', names=names, dtype=None, skiprows=1, usecols = usecols)
	#usecols=[70,73,76,79,82,85]) 

	dataset_noNan = dataset.dropna(axis=0)
	
	# Split-out validation dataset
	array = dataset_noNan.values
	X = array[:,0:-1]
	Y = array[:,-1]

	# Test options and evaluation metric
	seed = 7
	validation_size = 0.20
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

	# Reshape the various arrays
	np.reshape(X_train,-1)
	np.reshape(Y_train,-1)
	np.reshape(X_validation,-1)
	np.reshape(Y_validation,-1)
	flux2 = np.array(flux)

	# Make predictions on validation dataset with LinearRegression()
	LiR = LinearRegression()
	LiR.fit(X_train, Y_train)
	predictions = LiR.predict(X_validation)
	result = r2_score(Y_validation, predictions)
	print('Score: ', result)
	predictionfinal = LiR.predict([flux2,flux2])
	print('My flux prediction is:')
	print(predictionfinal[0])
	return predictionfinal[0]

	# Make predictions on validation dataset using PolynomialFeatures
	#model = make_pipeline(PolynomialFeatures(3), Ridge()) #LinearRegression()
	#model.fit(X_train, Y_train)
	#predictions = model.predict(X_validation)
	#print(r2_score(Y_validation, predictions))
	#predictionfinal = model.predict([flux2,flux2])
	#print('my flux prediction is:')
	#print(predictionfinal[0])

	# Plot predictions against real values
	#plt.scatter(Y_validation,predictions, s = 80, color='black', marker='s')
	#plt.plot([0,1000], [0,1000], color='grey')
	#plt.xscale('log') 
	#plt.yscale('log') 
	#plt.axis([1E-3,100,1E-3,100])
	#plt.show()	



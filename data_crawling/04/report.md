## Statistics from `/api/v1/statistics` endpoint

{
	"number_of_apartments": 16003,
	"number_of_params": 30,
	"property_stats": {
		"building_condition": {"null": 0,"number_of_types": 6,"нормальное": 729,"отличное": 1468,"требует": 77,"удовлетворительное": 52,"хорошее": 1515},
		"construction_year": {"max": 2020,"mean": 2012,"min": 1917,"std": 14},
		"floor": {"0": 54,"1": 1790,"10": 642,"11": 485,"12": 408,"13": 320,"14": 351,"15": 328,"16": 314,"17": 238,"18": 202,"19": 140,"2": 1842,"20": 182,
			"21": 142,"22": 136,"23": 140,"24": 93,"25": 65,"26": 8,"27": 5,"28": 3,"29": 1,"3": 1728,"30": 1,"32": 1,"33": 2,"35": 2,"4": 1420,"40": 1,"5": 1481,
			"6": 886,"7": 853,"8": 781,"9": 958,"number_of_types": 35},
		"heating": {"null": 0,"number_of_types": 4,"без отопления": 503,"индивидуальное": 2189,"централизованное": 13214},
		"price": {"max": 2900000,"mean": 73719,"min": 1475,"std": 94429},
		"price_uah": {"max": 74742280,"mean": 1899985,"min": 38022,"std": 2433751},
		"rooms_count": {"1": 5556,"10": 3,"2": 5220,"3": 4065,"4": 856,"5": 231,"6": 49,"7": 13,"8": 6,"9": 4,"number_of_types": 10},
		"seller": {"null": 0,"number_of_types": 6,"от застройщика": 245,"от посредника": 7294,"от представителя застройщика": 892,"от представителя хозяина (без комиссионных)": 1019,"от собственника": 319},
		"state_name": {"number_of_types": 20,"Винницкая": 1347,"Волынская": 22,"Днепропетровская": 163,"Житомирская": 65,"Закарпатская": 55,"Запорожская": 38,"Ивано-Франковская": 161,"Киевская": 3577,
			"Львовская": 67,"Николаевская": 70,"Одесская": 9544,"Полтавская": 14,"Ровенская": 60,"Тернопольская": 131,"Харьковская": 238,"Херсонская": 22,"Хмельницкая": 361,"Черкасская": 45,"Черновицкая": 23},
		"total_square_meters": {"max": 5996.0,"mean": 68,"min": 2.28,"std": 69},
		"wall_type": {"number_of_types": 24,"армированный железобетон": 3,"блочно-кирпичный": 159,"бутовый камень": 3,"газобетон": 118,"газоблок": 150,"дерево и кирпич": 5,"железобетон": 26,
			"инкерманский камень": 2,"каркасно-каменная": 6,"керамзитобетон": 94,"керамический блок": 22,"керамический кирпич": 1,"кирпич": 8758,"монолит": 1375,"монолитно-блочный": 104,
			"монолитно-каркасный": 891,"монолитно-кирпичный": 468,"монолитный железобетон": 59,"облицовочный кирпич": 1,"панель": 2053,"пеноблок": 762,"ракушечник (ракушняк)": 871,"сборно-монолитная": 2,"силикатный кирпич": 70},
		"water": {"null": 0,"number_of_types": 5,"колодец": 3,"скважина": 9,"централизованное (водопровод)": 3732,"централизованное (водопровод) • скважина": 6}
	},
	"state_price_stats": {
		"Винницкая": {"max": 1306000,"mean": 43246,"min": 9809,"std": 41971.29165385},
		"Волынская": {"max": 150000,"mean": 50346,"min": 19670,"std": 36857.21632995},
		"Днепропетровская": {"max": 350000,"mean": 63406,"min": 8500,"std": 56951.25510289},
		"Житомирская": {"max": 1219344,"mean": 63237,"min": 14500,"std": 169092.62829958},
		"Закарпатская": {"max": 90000,"mean": 52471,"min": 20000,"std": 18736.92453850},
		"Запорожская": {"max": 95000,"mean": 41883,"min": 10000,"std": 20625.88475437},
		"Ивано-Франковская": {"max": 660000,"mean": 37626,"min": 13117,"std": 53289.34690475},
		"Киевская": {"max": 2500000,"mean": 107890,"min": 1475,"std": 135429.39315254},
		"Львовская": {"max": 480000,"mean": 70330,"min": 17460,"std": 75834.18444476},
		"Николаевская": {"max": 185000,"mean": 50595,"min": 13000,"std": 30047.44491942},
		"Одесская": {"max": 2900000,"mean": 69709,"min": 6800,"std": 81537.31644241},
		"Полтавская": {"max": 186290,"mean": 70643,"min": 16000,"std": 52637.51508151},
		"Ровенская": {"max": 65800,"mean": 32589,"min": 12800,"std": 12945.59214178},
		"Тернопольская": {"max": 99500,"mean": 39421,"min": 10800,"std": 16901.13794033},
		"Харьковская": {"max": 300820,"mean": 57109,"min": 10500,"std": 45826.97446258},
		"Херсонская": {"max": 75500,"mean": 33409,"min": 18000,"std": 13544.91470372},
		"Хмельницкая": {"max": 100000,"mean": 28569,"min": 11280,"std": 13038.74676835},
		"Черкасская": {"max": 470000,"mean": 50215,"min": 14000,"std": 74918.24375775},
		"Черновицкая": {"max": 75000,"mean": 37437,"min": 14000,"std": 14227.17974630}
	}
}

## Metric(MAE/MSE), loss and inference time for DecisionTree and LightGBM models

DecisionTree:
- test MAE: 12477
- train MAE: 11249
- test MSE: 306025215
- train MSE: 253544448
- inference time: 0.0011050701141357422

XGBoost:
- train MAE: 8639.66673429991
- test MAE: 11567.456151077315
- train MSE: 153244192.00643247
- test MSE: 271019471.7461254
- inference time: 0.007146120071411133

## Metric(MAE/MSE), loss and inference time for Feed-Forward NN for each set of hyperparams you tried

hidden_dim=25, hidden_num=3, activation_function='relu':
- train MAE: 12268.726234375
- test MAE: 12852.46484375
- train MSE: 291503962.88
- test MSE: 335385544.0
- inference time: 0.0006666183471679688

hidden_dim=10, hidden_num=3, activation_function='relu':
- train MAE: 12731.2619609375
- test MAE: 13335.09828404018
- train MSE: 299008441.728
- test MSE: 340756146.28571427
- inference time: 0.00036025047302246094

hidden_dim=50, hidden_num=3, activation_function='relu':
- train MAE: 12105.99203125
- test MAE: 12742.804966517857
- train MSE: 288735902.336
- test MSE: 339590480.0
- inference time: 0.00039196014404296875

hidden_dim=71, hidden_num=3, activation_function='relu':
- train MAE: 11924.275796875
- test MAE: 12547.038295200893
- train MSE: 288076790.4
- test MSE: 342342541.71428573
- inference time: 0.0003542900085449219

hidden_dim=25, hidden_num=2, activation_function='relu':
- train MAE: 12582.1843046875
- test MAE: 13162.546595982143
- train MSE: 296072510.976
- test MSE: 338240777.14285713
- inference time: 0.00029659271240234375

hidden_dim=25, hidden_num=5, activation_function='relu':
- train MAE: 12068.1057890625
- test MAE: 12560.060337611607
- train MSE: 299314896.512
- test MSE: 346707725.71428573
- inference time: 0.0004730224609375

hidden_dim=25, hidden_num=10, activation_function='relu':
- train MAE: 12189.632578125
- test MAE: 13010.055454799107
- train MSE: 285157354.88
- test MSE: 341639259.4285714
- inference time: 0.0009317398071289062

hidden_dim=25, hidden_num=3, activation_function='leaky':
- train MAE: 12463.9586953125
- test MAE: 13029.8115234375
- train MSE: 298054769.28
- test MSE: 339592056.0
- inference time: 0.0003497600555419922

hidden_dim=25, hidden_num=3, activation_function='elu':
- train MAE: 12167.2282109375
- test MAE: 12702.176060267857
- train MSE: 289475403.776
- test MSE: 331444429.71428573
- inference time: 0.0003864765167236328

## All hyperparams (K-fold/Cross-validation, size of train and test datasets, percentage for holdout)

- k-fold/cross_validation: n_splits=5, shuffle=True, scoring='neg_mean_absolute_error'
- train size: 12485, test size: 1388 (10%), 1000 rows were deleted as outliers
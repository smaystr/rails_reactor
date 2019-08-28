{"flat_statistics":{
	"avg_kitchen_area":10,
	"avg_living_area":122,
	"avg_total_area":63,
	"max_kitchen_area":196,
	"max_living_area":362615,
	"max_total_area":23000,
	"min_kitchen_area":0,
	"min_living_area":0,
	"min_total_area":6
},
"price_statistics":{
	"avg_price_UAH":1339768,
	"avg_price_USD":52689,
	"max_price_UAH":88832485,
	"max_price_USD":3500000,
	"min_price_UAH":382,
	"min_price_USD":15
},
"table_column_names_announcement_info":["announcement_info_flat_id","announcement_info_page_url","announcement_info_title","announcement_info_price_uah","announcement_info_price_usd","announcement_info_image_urls","announcement_info_description","announcement_info_type_of_proposal","announcement_info_verified","announcement_info_date_created"],

"table_column_names_flat_info":["flat_info_flat_id","flat_info_street_name","flat_info_city_name","flat_info_total_area","flat_info_living_area","flat_info_kitchen_area","flat_info_floor","flat_info_total_number_of_floors","flat_info_number_of_rooms","flat_info_year_of_construction","flat_info_heating_type","flat_info_walls_type","flat_info_latitude","flat_info_longitude"],

"table_names":["announcement_info","flat_info"],

"total_number_of_flats_in_DB":30011}

## Metrics MSE/MAE, inference for Decision Tree and CatBoost
Decision Tree:
- inference time 0.28339195251464844
- MSE train - 14,935,157  MSE test - 977,485,431
- MAE train - 1,525      MAE test - 13,459

CatBoost 
- inference time 13.471676111221313 as I used 10 iterations 
- MSE train - 243,448,105   MSE test - 854,246,223
- MAE train - 9,126 		  MAE test - 13,699

## Metrics MSE/MAE, inference for Neural Network
 * 2 hidden layers with 256 units; activation: relu
 - MSE train - 4,779,152,400  MSE test - 5,099,856,000
 - MAE train - 45,959 		  MAE test - 46,631

 * 3 hidden layers with 256 units; activation: relu
 - MSE train - 2,735,021,800  MSE test - 2,967,771,400
 - MAE train - 31,766 		  MAE test - 31,988

 * 4 hidden layers with 256 units; activation: relu
 - MSE train - 2,682,088,200  MSE test - 2,928,655,400
 - MAE train - 30,767 		  MAE test - 30,947

 * 2 hidden layers with 512 units; activation: relu
 - MSE train - 3,560,478,200  MSE test - 3,844,341,000
 - MAE train - 33,123 		  MAE test - 33,747

 * 3 hidden layers with 512 units; activation: relu
 - MSE train - 2,797,349,000  MSE test - 3,039,971,000
 - MAE train - 33,045 		  MAE test - 33,212

 * 4 hidden layers with 512 units; activation: relu
 - MSE train - 2,714,086,400  MSE test - 2,964,049,400
 - MAE train - 29,017		  MAE test - 29,301

## All hyperparameters 
total dataset ≈ 30000
- train_test_split: test_size = 0.2, shuffle=True, random_state=42
- around 1000 rows were deleted as outliers
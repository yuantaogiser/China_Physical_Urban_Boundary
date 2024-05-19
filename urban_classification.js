// var image = ee.Image('projects/ee-yuantaogiser/assets/Raster/hefei_2020')
var image = ee.Image('projects/ee-yuantaogiser/assets/Raster/anhui_isad_2021')


// var target_area = '合肥市'
// var table = ee.FeatureCollection("users/yuantaogiser/China_shi").filter(ee.Filter.eq('市',target_area))
var target_area = 'anhui'
var table = ee.FeatureCollection("projects/ee-yuantaogiser/assets/Vector/China_province").filter(ee.Filter.eq('name',target_area))
var boundary = ee.FeatureCollection('projects/ee-yuantaogiser/assets/Vector/ChinaUrbanBoundary')
Map.addLayer(boundary)

//--------------------- NTL --------------------------------------------

//—————————————————————————— NightTime Light  ——————————————————————————
var VIIRS_image = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').select(['avg_rad'],['ntl'])
var DMSP_image = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS").select(['avg_vis'],['ntl'])

var scale_VIIRS = VIIRS_image.filterDate('2012-04-01', '2014-01-01').median()
var scale_DMSP  = DMSP_image.filterDate('2012-04-01', '2014-01-01').median()
var scale_factor = scale_VIIRS.divide(scale_DMSP)

var scaling_image = function(image){
  return image.multiply(scale_factor)
}

var VIIRS = VIIRS_image.filterDate('2012-01-01', '2021-12-31')
var DMSP  = DMSP_image.filterDate('1991-01-01', '2014-12-31').map(scaling_image)

var ntl_merge = VIIRS.merge(DMSP).max()
//—————————————————————————— NightTime Light  ——————————————————————————


var targetImage =image.clip(table).addBands(ntl_merge)
Map.addLayer(targetImage)
// Map.addLayer(median_VIIRS)

// var bands = ['b1','ntl'];
var bands = ['b1','ntl'];

// var train = samples_urban.merge(samples_rural);
var train = ee.FeatureCollection('projects/ee-yuantaogiser/assets/Vector/hefeiUrbanOrNot')

var training = targetImage.select(bands).sampleRegions({
  collection: train,
  properties: ['urbanOrNot'],
  scale: 30,
  tileScale :16
});
  

var withRandom = training.randomColumn('random');

var split = 0.7; // Avoid over-fit
var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));
var testingPartition = withRandom.filter(ee.Filter.gte('random', split));

// var classifier = ee.Classifier.smileRandomForest(10).train({
//   features: trainingPartition,
//   classProperty: 'landcover',
//   inputProperties: bands
// });
var classifier = ee.Classifier.libsvm().train({  
  features: trainingPartition,
  classProperty: 'urbanOrNot',
  inputProperties: bands
});
// var classifier = ee.Classifier.libsvm({svmType: 'ONE_CLASS'}).train({  
//   features: trainingPartition,
//   classProperty: 'landcover',
//   inputProperties: bands
// });
//-----------------------------------------------------------------

var classified = targetImage.select(bands).classify(classifier);

var test = testingPartition.classify(classifier);
 
var confusionMatrix = test.errorMatrix('urbanOrNot', 'classification');
print('confusionMatrix',confusionMatrix);
print('overall accuracy', confusionMatrix.accuracy());
print('kappa accuracy', confusionMatrix.kappa());

Map.addLayer(classified,{min: 0, max: 4, palette: ['red', 'green', 'blue','yellow']}, 'classification', false);

Map.addLayer(classified.selfMask(),{min: 0, max: 4, palette: ['green','red', 'blue','yellow']}, 'classification');

// Export.image.toDrive({
//         image: classified.selfMask(),
//         description: 'urban_beijing',
//         // crs: "EPSG:32650",
//         scale: 30,
//         region: table2.geometry(), 
//         maxPixels: 1e13,
//         folder: 'myExport'
//       });
      

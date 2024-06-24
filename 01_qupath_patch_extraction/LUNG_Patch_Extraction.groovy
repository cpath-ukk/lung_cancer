import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

pathOutputGlob='/Users/drpusher/work/46_LUNG_WORKING/01_TEST_UKK/01_DS_RAW/01_DS_TEST_UKK_TUMOR_COLOR/02_LUSC/' 

// Define output path (here, relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(pathOutputGlob, name)
mkdirs(pathOutput)

// Define output path (relative to project)
def pathOutput1= buildFilePath(pathOutput, 'ALL')
mkdirs(pathOutput1)

// Define output resolution
double requestedPixelSize = 1.0
int tile_size = 512
exte = '.jpg'
int overl = 0

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()


// Create an ImageServer where the pixels are derived from annotations
def labelServer1 = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('TU_LUAD', 1)      // Choose output labels (the order matters!)
    .addLabel('TU_LUAD_MUC', 1)
    .addLabel('TU_LEPID', 1)
    .addLabel('TU_ART_LUAD', 1)
    .addLabel('TU_LUSC', 1)
    .addLabel('TU_ART_LUSC', 1)
    .addLabel('TU_UNCLEAR', 1)
    .addLabel('TU_STROMA', 2)
    .addLabel('RESORPTION', 2)
    .addLabel('RESORP', 2)
    .addLabel('NECROSIS', 3)
    .addLabel('HORN', 3)
    .addLabel('MUCUS', 4)
    .addLabel('MUCIN', 4)
    .addLabel('BENIGN', 5)
    .addLabel('BENIGN_ETC', 5)
    .addLabel('BENIGN_PEC', 5)
    .addLabel('STR', 6)
    .addLabel('NERVE', 6)
    .addLabel('BLOOD', 7)
    .addLabel('ERYTHROCYTES', 7)
    .addLabel('BRONCHUS', 8)
    .addLabel('CARTIL', 9)
    .addLabel('CARTILAGE', 9)
    .addLabel('FAT', 6)
    .addLabel('GLAND_BRONCH', 10)
    .addLabel('INFLAM', 11)
    .addLabel('LYM_AGGR', 11)
    .addLabel('LYMPH_NODE', 11)
    .addLabel('MUSCLE', 6)
    .addLabel('VESSEL', 6)
    //.addLabel('BACK', 12)
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension(exte)     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(tile_size)              // Define size of each tile, in pixels
    .labeledServer(labelServer1) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(overl)                // Define overlap, in pixel units at the export resolution
    //.includePartialTiles(true)
    .writeTiles(pathOutput1)     // Write tiles to the specified directory

print 'Done for ALL!'

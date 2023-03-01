import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// CHANGE THESE PARAMETERS

// DEFINE TILE SIZE HERE
int tileSize = 512


// DEFINE OUTPUT RESOLUTION
double requestedPixelSize = .2525

// DEFINE OUTPUT PATH (RELATIVE TO PROJECT)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'training_tiles')
mkdirs(pathOutput)




// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()



//ADD CLASS LABELS WITH THE .addLabel() function.

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0) // Specify background label (usually 0 or 255)
    .grayscale(true)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Tubules', 1)      // Choose output labels (the order matters!)
    .addLabel('Gloms', 2)
    .addLabel('Vessels',3)
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()




// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(tileSize)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .imageSubDir('imgs')
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print 'Done!'
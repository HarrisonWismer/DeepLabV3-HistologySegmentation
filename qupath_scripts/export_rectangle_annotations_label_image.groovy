import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'export', name)
mkdirs(pathOutput)

// Define output resolution
double requestedPixelSize = 0.5077

// Convert to downsample
double downsample = 1.0

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Vessels',1)      // Choose output labels (the order matters!)
    .addLabel('Tubules',2)
    .addLabel('Gloms',3)
    .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
    .build()



// Export each region
int i = 0
for (annotation in getAnnotationObjects()) {
    if (annotation.ROI.toString().contains('Rectangle')) {
        def region = RegionRequest.createInstance(
            labelServer.getPath(), downsample, annotation.getROI())
        i++
        def outputPath = buildFilePath(pathOutput, annotation.Name+'.png')
        writeImageRegion(labelServer, region, outputPath)
        }
}
package rapid.try5;

import com.flir.flironesdk.RenderedImage;

interface savingThreads extends Runnable {
    void run(RenderedImage renderedImage, String formatDate);
}

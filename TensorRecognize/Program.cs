using System.IO;
using TensorFlow;
using System.Collections.Generic;

namespace TensorRecognize
{
    class ProcessTensorflow
    {
        static TFGraph graph = null;

        public static void Main()
        {
            string name = "tools";
            string recognizerGraph = @"G:\Tensorflow-Inception\trainedCNN\" + name + "_recognizer.pb";
            string recognizerLabels = @"G:\Tensorflow-Inception\trainedCNN\" + name + "_labels.lst";
            string filename = "c:\\tmp\\temp.png";

            List<string> files = new List<string>() { filename };

            // Construct an in-memory graph from the serialized form.
            if (graph == null)
            {
                graph = new TFGraph();
                // Load the serialized GraphDef from a file.
                var model = File.ReadAllBytes(recognizerGraph);
                graph.Import(model, "");
            }
            TensorAnalysis.Process(graph, recognizerLabels, filename);
        }
    }
}

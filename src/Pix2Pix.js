let tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');



class DataLoader {
    // class DataLoader():
    constructor(rootPath, params) {

        // TODO: Error on no root location provided or set a default once server is up
        this.rootPath = rootPath;
        // defaultParams = {
        //     location: 'yosemite',
        //     patchSize: 256,
        //     scale: 7,
        //     dataset: 'train',
        //     channels: [['topo_grid8'],
        //                ['topo']],
        //     numSamples: 400,
        //     dataType: '.jpg'
        // }

        // DATSET SELECTION PARAMS
        this.location = (params.location !== undefined) ? params.location : 'yosemite';
        this.patchSize = (params.patchSize !== undefined) ? params.patchSize : 256;
        this.scale = (params.scale !== undefined) ? params.scale : 8;
        this.dataset = (params.dataset !== undefined) ? params.dataset : 'train';
        this.channels = (params.channels !== undefined) ? params.channels : [['grid_8_bin'], ['topo']];
        this.numSamples = (params.numSamples !== undefined) ? params.numSamples : 400;
        this.dataType = (params.dataType !== undefined) ? params.dataType : '.png';

        this.datasetPath = this.rootPath + '/' + this.location + '/' + this.patchSize + '/' + this.scale + '/' + this.dataset + '/';

        this.numChannelsA = this.channels[0].length;
        this.numChannelsB = this.channels[1].length;

        // Construct Paths for Input A (Truth) and Input B (Representation)
        this.datasetPathsA = [];
        this.datasetPathsB = [];

        for (let i = 1; i <= this.numSamples; i++) {

            let inputChannelPathsA = [];
            for (let channelNameA of this.channels[0]) {
                inputChannelPathsA.push(this.datasetPath + channelNameA + '/' + i + this.dataType);
            }
            this.datasetPathsA.push(inputChannelPathsA);

            let inputChannelPathsB = [];
            for (let channelNameB of this.channels[1]) {
                inputChannelPathsB.push(this.datasetPath + channelNameB + '/' + i + this.dataType);
            }
            this.datasetPathsB.push(inputChannelPathsB);
        }

    }

    async loadChannel(path) {
        const image = await Jimp.read(path);
        const imageArray = new Array(image.bitmap.width * image.bitmap.height);
        const offset = 127.5;
        for (let i = 0, j = 0; i < imageArray.length; i++) {
            imageArray[i] = image.bitmap.data[j] / offset - 1.0;
            j += 4
        }

        return imageArray;
    }

    async loadChannels(paths) {
        // console.log(paths);
        const numChannels = paths.length;
        const images = []

        // Get all of the image channels
        for (let i = 0; i < numChannels; i++) {
            images[i] = await Jimp.read(paths[i]);
        }

        // console.log(images);


        const size = images[0].bitmap.width * images[0].bitmap.width;
        const offset = 127.5; // ?? This is for 8-bit offset

        // Normalize and interleave image channels
        const channelArray = new Array(size * numChannels);
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < numChannels; j++) {
                channelArray[numChannels * i + j] = images[j].bitmap.data[i * 4] / offset - 1.0;
            }
        }

        // console.log(channelArray);

        return channelArray;
    }

    //  DATASET PATH                            (root URL to Datasets)
    //  |---datasets.json                       (JSON describing dataset)
    //  |---DATASET NAME                        (location name?)
    //  |   |---.rawdata                        (blob of raw data for dataset)
    //  |   |   |---nw034.img                   (random data blob)
    //  |   |   |---nw035.img                   (random data blob)
    //  |   |---<DATASET TYPE>                  (train/test/val)
    //  |   |   |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |   |   |   |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |   |   |   |---2.jpg
    //  |   |   |   |---3.jpg
    //  |   |   |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |   |       |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |   |       |---2.jpg
    //  |   |       |---3.jpg
    //  |   |---<DATASET TYPE>                  (train/test/val)
    //  |       |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |       |   |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |       |   |---2.jpg
    //  |       |   |---3.jpg
    //  |       |---<DATASET CHANNEL/S NAME>    (bitmap[1], rgb[3], rgba[4]);
    //  |           |---1.jpg                   (numbers should pair across all channels per name and type)
    //  |           |---2.jpg
    //  |           |---3.jpg
    //  .scripts                            (python scripts for generating data patches)
    //  |---genDataset.py                   (pyhton script)

    async loadBatch(batchSize = 1) {

        let batchArrayA = [];
        let batchArrayB = [];

        for (let b = 0; b < batchSize; b++) {
            let index = Math.floor(Math.random() * this.numSamples);
            let nextA = await this.loadChannels(this.datasetPathsA[index]);
            let nextB = await this.loadChannels(this.datasetPathsB[index]);

            batchArrayA = batchArrayA.concat(nextA);
            batchArrayB = batchArrayB.concat(nextB);

            // console.log(batchArrayA.length);

        }

        return [batchArrayA, batchArrayB];
    }

}

// Pix2Pix Generator
class Pix2Pix {
    constructor(imageShape, numFilters, rootPath, params) {
        // # Input shape
        // self.img_rows = 256
        // self.img_cols = 256
        // self.channels = 3
        // self.img_shape = (self.img_rows, self.img_cols, self.channels)
        this.imageShape = imageShape; // ex: [256, 256, 3] = [rows, columns, channels]
        this.imageRows = imageShape[0];
        this.imageColumns = imageShape[1];
        this.imageChannels = imageShape[2];

        // # Configure data loader
        // self.dataset_name = 'facades'
        // self.data_loader = DataLoader(dataset_name = self.dataset_name,
        this.dataLoader = new DataLoader(rootPath, params);
        // defaultParams = {
        //     location: 'yosemite',
        //     patchSize: 256,
        //     scale: 7,
        //     dataset: 'train',
        //     channels: [['topo_grid8'],
        //                ['topo']],
        //     numSamples: 400,
        //     dataType: '.jpg'
        // }




        // img_res = (self.img_rows, self.img_cols))


        // # Calculate output shape of D(PatchGAN)
        // patch = int(self.img_rows / 2 ** 4)
        // self.disc_patch = (patch, patch, 1)
        // this.dPatch = Math.floor(this.imageRows / Math.pow(2, 4)); // ?? What is this?
        this.dPatch = Math.floor(this.imageRows);


        // # Number of filters in the first layer of G and D
        // self.gf = 64
        // self.df = 64
        this.g_filters = numFilters;
        this.d_filters = numFilters;

        // optimizer = Adam(0.0002, 0.5)
        this.learningRate = 0.0002;
        this.beta1 = 0.5;
        this.beta2 = 0.5;
        this.epsilon = 0.5;
        this.adam = tf.train.adam(this.learningRate, this.beta1);

        // # Build and compile the discriminator
        // self.discriminator = self.build_discriminator()
        // self.discriminator.compile(loss = 'mse',
        //     optimizer = optimizer,
        //     metrics = ['accuracy'])
        this.discriminator = this.buildDiscriminator();
        this.discriminator.compile({
            optimizer: this.adam,
            loss: 'meanSquaredError',
            metrics: ['accuracy']
        });
        this.discriminator.summary();
        // #-------------------------
        // # Construct Computational
        // #   Graph of Generator
        // #-------------------------

        // # Build the generator
        // self.generator = self.build_generator()
        this.generator = this.buildGenerator();

        // # Input images and their conditioning images
        // img_A = Input(shape = self.img_shape)
        // img_B = Input(shape = self.img_shape)
        const inputA = tf.input({ shape: this.imageShape });
        const inputB = tf.input({ shape: this.imageShape });

        // # By conditioning on B generate a fake version of A
        // fake_A = self.generator(img_B)
        const fakeA = this.generator.apply(inputB); //????

        // # For the combined model we will only train the generator
        // self.discriminator.trainable = False
        this.discriminator.trainable = false;

        // # Discriminators determines validity of translated images / condition pairs
        // valid = self.discriminator([fake_A, img_B])
        const valid = this.discriminator.apply([fakeA, inputB]); //????
        // const valid = this.discriminator.getLayer('conv2d_Conv2D5');

        // self.combined = Model(inputs = [img_A, img_B], outputs = [valid, fake_A])
        this.combined = tf.model({ inputs: [inputA, inputB], outputs: [valid, fakeA] });

        // self.combined.compile(loss = ['mse', 'mae'],
        //     loss_weights = [1, 100],
        //     optimizer = optimizer)
        this.combined.summary();
        this.combined.compile({
            optimizer: this.adam,
            loss: ['meanSquaredError', 'meanAbsoluteError'],
            lossWeights: [1, 100]
        });
    }

    buildGenerator() {
        // def conv2d(layer_input, filters, f_size = 4, bn = True):
        // """Layers used during downsampling"""
        // d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = 'same')(layer_input)
        // d = LeakyReLU(alpha = 0.2)(d)
        // if bn:
        //     d = BatchNormalization(momentum = 0.8)(d)
        // return d
        function conv2d(layer_input, filters, f_size = 4, bn = true) {
            let d = tf.layers.conv2d({
                kernelSize: f_size,
                filters: filters,
                strides: 2,
                // activation: 'relu',
                // kernelInitializer: 'VarianceScaling', 
                padding: 'same'
            }).apply(layer_input);

            d = tf.layers.leakyReLU({
                alpha: 0.2
            }).apply(d);

            if (bn) d = tf.layers.batchNormalization({ momentum: 0.8 }).apply(d);

            return d;
        }

        // def deconv2d(layer_input, skip_input, filters, f_size = 4, dropout_rate = 0):
        // """Layers used during upsampling"""
        // u = UpSampling2D(size = 2)(layer_input)
        // u = Conv2D(filters, kernel_size = f_size, strides = 1, padding = 'same', activation = 'relu')(u)
        // if dropout_rate:
        //     u = Dropout(dropout_rate)(u)
        // u = BatchNormalization(momentum = 0.8)(u)
        // u = Concatenate()([u, skip_input])
        // return u

        function deconv2d(layer_input, skip_input, filters, f_size = 4, dropout_rate = 0) {
            let u = tf.layers.upSampling2d({
                size: [2, 2]
            }).apply(layer_input);

            u = tf.layers.conv2d({
                kernelSize: f_size,
                filters: filters,
                strides: 1,
                activation: 'relu',
                // kernelInitializer: 'VarianceScaling', 
                padding: 'same'
            }).apply(u);

            if (dropout_rate) u = tf.layers.dropout({
                rate: dropout_rate
            }).apply(u);

            u = tf.layers.batchNormalization({ momentum: 0.8 }).apply(u);

            u = tf.layers.concatenate().apply([u, skip_input]);

            return u;
        }

        // # Image input
        // d0 = Input(shape = self.img_shape)
        const d0 = tf.input({ shape: this.imageShape });

        // # Downsampling
        // d1 = conv2d(d0, self.gf, bn = False)
        // d2 = conv2d(d1, self.gf * 2)
        // d3 = conv2d(d2, self.gf * 4)
        // d4 = conv2d(d3, self.gf * 8)
        // d5 = conv2d(d4, self.gf * 8)
        // d6 = conv2d(d5, self.gf * 8)
        // d7 = conv2d(d6, self.gf * 8)

        const d1 = conv2d(d0, this.g_filters, 4, false);
        const d2 = conv2d(d1, this.g_filters * 2);
        const d3 = conv2d(d2, this.g_filters * 4);
        const d4 = conv2d(d3, this.g_filters * 8);
        const d5 = conv2d(d4, this.g_filters * 8);
        const d6 = conv2d(d5, this.g_filters * 8);
        const d7 = conv2d(d6, this.g_filters * 8);


        // # Upsampling
        // u1 = deconv2d(d7, d6, self.gf * 8)
        // u2 = deconv2d(u1, d5, self.gf * 8)
        // u3 = deconv2d(u2, d4, self.gf * 8)
        // u4 = deconv2d(u3, d3, self.gf * 4)
        // u5 = deconv2d(u4, d2, self.gf * 2)
        // u6 = deconv2d(u5, d1, self.gf)

        const u1 = deconv2d(d7, d6, this.g_filters * 8);
        const u2 = deconv2d(u1, d5, this.g_filters * 8);
        const u3 = deconv2d(u2, d4, this.g_filters * 8);
        const u4 = deconv2d(u3, d3, this.g_filters * 4);
        const u5 = deconv2d(u4, d2, this.g_filters * 2);
        const u6 = deconv2d(u5, d1, this.g_filters);

        // u7 = UpSampling2D(size = 2)(u6)
        // output_img = Conv2D(self.channels, kernel_size = 4, strides = 1, padding = 'same', activation = 'tanh')(u7)
        const u7 = tf.layers.upSampling2d({ size: [2, 2] }).apply(u6);
        const output = tf.layers.conv2d({
            kernelSize: 4,
            filters: this.imageChannels,
            strides: 1,
            activation: 'tanh',
            // kernelInitializer: 'VarianceScaling', 
            padding: 'same'
        }).apply(u7);

        // return Model(d0, output_img)
        return tf.model({ inputs: d0, outputs: output });


    }

    buildDiscriminator() {
        // def d_layer(layer_input, filters, f_size = 4, bn = True):
        // """Discriminator layer"""
        // d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = 'same')(layer_input)
        // d = LeakyReLU(alpha = 0.2)(d)
        // if bn:
        //     d = BatchNormalization(momentum = 0.8)(d)
        // return d

        function dLayer(layer_input, filters, f_size = 4, bn = true) {
            let d = tf.layers.conv2d({
                kernelSize: f_size,
                filters: filters,
                strides: 2,
                // activation: 'relu',
                // kernelInitializer: 'VarianceScaling', 
                padding: 'same'
            }).apply(layer_input);

            d = tf.layers.leakyReLU({
                alpha: 0.2
            }).apply(d);

            if (bn) d = tf.layers.batchNormalization({ momentum: 0.8 }).apply(d);

            return d;
        }

        // img_A = Input(shape = self.img_shape)
        // img_B = Input(shape = self.img_shape)
        let input1 = tf.input({ shape: this.imageShape });
        let input2 = tf.input({ shape: this.imageShape });

        // # Concatenate image and conditioning image by channels to produce input
        // combined_imgs = Concatenate(axis = -1)([img_A, img_B])
        let combinedImgs = tf.layers.concatenate().apply([input1, input2]);

        // d1 = d_layer(combined_imgs, self.df, bn = False)
        // d2 = d_layer(d1, self.df * 2)
        // d3 = d_layer(d2, self.df * 4)
        // d4 = d_layer(d3, self.df * 8)

        const d1 = dLayer(combinedImgs, this.d_filters);
        const d2 = dLayer(d1, this.d_filters * 2);
        const d3 = dLayer(d2, this.d_filters * 4);
        const d4 = dLayer(d3, this.d_filters * 8);

        // validity = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(d4)
        const output = tf.layers.conv2d({
            // inputShape: [28, 28, 1],
            kernelSize: 4,
            filters: 1,
            strides: 1,
            padding: 'same'
        }).apply(d4);

        // return Model([img_A, img_B], validity)
        return tf.model({ inputs: [input1, input2], outputs: output });
    }

    async trainBatch(batchSize = 1) {
        // start_time = datetime.datetime.now()
        // let fakeA, dLossReal, dLossFake, dLoss, gLoss;
        let batch = await this.dataLoader.loadBatch(batchSize);
        let batchA = batch[0];
        // console.log(batchA);
        let batchB = batch[1];

        // return tf.tidy(async () => {
        const valid = tf.ones([batchSize, this.dPatch, this.dPatch, 1]);
        const fake = tf.zeros([batchSize, this.dPatch, this.dPatch, 1]);

        const shapeA = [batchSize, this.dataLoader.patchSize, this.dataLoader.patchSize, this.dataLoader.numChannelsA];
        const shapeB = [batchSize, this.dataLoader.patchSize, this.dataLoader.patchSize, this.dataLoader.numChannelsB];
        // console.log(shapeA);

        // Input 4D Tensors
        const inputA = tf.tensor4d(batchA, shapeA);
        const inputB = tf.tensor4d(batchB, shapeB)

        // Train Discriminator
        // ---------------------

        // Condition on B and generate a translated version
        const fakeA = this.generator.predictOnBatch(inputB);
        // console.log(inputA);
        // console.log(fakeA);
        // console.log(valid);
        

        // Train the discriminators(original images = real / generated = Fake)
        const dLossReal = await this.discriminator.trainOnBatch([inputA, inputB], valid);
        console.log(dLossReal);
        
        const dLossFake = await this.discriminator.trainOnBatch([fakeA, inputB], fake);
        console.log(dLossFake);

        const dLoss = 0.5 * tf.add(dLossReal, dLossFake); // TODO: FIX THIS
        console.log(dLoss);

        // Train Generator
        // ---------------------

        // Train the generators
        const gLoss = await this.combined.trainOnBatch([inputA, inputB], [valid, inputA]);
        console.log(gLoss);
        

        console.log('[D loss: ' + dLoss[0] + ', acc: ' + (100 * dLoss[1]) + '] [G loss: ' + gLoss[0] + ']');
        // return {disLoss: dLoss, genLoss: gLoss}


        // });
    }

    async train(data, epochs, batchSize = 1, sampleInterval = 50) {
        // start_time = datetime.datetime.now()
        let fakeA, dLossReal, dLossFake, dLoss, gLoss;

        // # Adversarial loss ground truths
        // valid = np.ones((batch_size,) + self.disc_patch)
        // fake = np.zeros((batch_size,) + self.disc_patch)
        const valid = tf.ones([batchSize, this.dPatch, this.dPatch, 1]); //??
        const fake = tf.zeros([batchSize, this.dPatch, this.dPatch, 1]); //??


        // for epoch in range(epochs):
        //     for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
        for (let epoch in epochs) {
            let batch = await this.dataLoader.loadBatch(batchSize);

            let batchA = batch[0];
            let batchB = batch[1];
        }

        // # ---------------------
        // #  Train Discriminator
        // # ---------------------

        // # Condition on B and generate a translated version
        // fake_A = self.generator.predict(imgs_B)
        fakeA = this.generator.predict(imgPair.b);

        // # Train the discriminators(original images = real / generated = Fake)
        // d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        // d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
        // d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        dLossReal = await this.discriminator.trainOnBatch([imgPair.a, imgPair.b], valid);
        dLossFake = await this.discriminator.trainOnBatch([fakeA, imgPair.b], fake);
        dLoss = 0.5 * tf.addStrict(dLossReal, dLossFake);

        // # -----------------
        // #  Train Generator
        // # -----------------

        // # Train the generators
        // g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
        gLoss = await this.combined.trainOnBatch([imgPair.a, imgPair.b], [valid, imgPair.a]);

        // elapsed_time = datetime.datetime.now() - start_time
        // # Plot the progress
        console.log('[Epoch ' + epoch + '/' + epochs + '] [Batch ' + batch_i + '/' + self.data_loader.n_batches + '] [D loss: ' + d_loss[0] + ', acc: ' + (100 * d_loss[1]) + '] [G loss: ' + g_loss[0] + '] time: ' + elapsed_time);
        // # If at save interval => save generated image samples
        // if batch_i % sample_interval == 0:
        //     self.sample_images(epoch, batch_i)

        if (batch_i % sampleInterval == 0) {
        }

    }

    static preprocess(image) {
        return tf.tidy(() => {

            //convert to a tensor 
            const tensor = tf.browser.fromPixels(image).toFloat();

            //resize 
            const resized = tf.image.resizeBilinear(tensor, [256, 256]);

            //normalize 
            const offset = tf.scalar(127.5);
            const normalized = resized.div(offset).sub(tf.scalar(1.0));

            //We add a dimension to get a batch shape 
            // const batched = normalized.expandDims(0);
            return normalized;

        });
    }

    static postprocess(tensor, w, h) {
        return tf.tidy(() => {

            //normalization factor  
            const scale = tf.scalar(0.5);

            //unnormalize and sqeeze 
            const squeezed = tensor.squeeze().mul(scale).add(scale);

            //resize to canvas size 
            const resized = tf.image.resizeBilinear(squeezed, [w, h]);

            return resized;
        });
    }

    predict(image) {
        return tf.tidy(() => {

            //get the prediction
            const gImg = model.predict(preprocess(imgData));

            //post process
            const postImg = postprocess(gImg, 512, 512);
            return postImg;

        });
    }
}



//---------------
// Main Test
//---------------
console.log(tf.getBackend());


// NN summary
// ------------------
const epochs = 1;
const imageShape = [256, 256, 1];
const numFilters = 32
const rootPath = './img'

const params = {
    location: 'yosemite',
    patchSize: 256,
    scale: 8,
    dataset: 'train',
    channels: [['topo'], ['grid_8_bin']],
    numSamples: 1000,
    dataType: '.png'
}

let p2p = new Pix2Pix(imageShape, numFilters, rootPath, params);
// p2p.discriminator.summary();
// p2p.generator.summary();

// console.log(p2p.dataLoader.datasetPathsA[0].length);
for (let i = 0; i < epochs; i++) {
    p2p.trainBatch(1);
}

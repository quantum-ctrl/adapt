import * as Physics from './physics.js';

export class KConverter {
    static convert2D(data, meta, hv, workFunc) {
        console.log("KConverter: Converting 2D to k-space...");
        const angleAxis = meta.axes.kx; // Angle
        const energyAxis = meta.axes.energy; // Energy
        const width = meta.data_info.shape[1];
        const height = meta.data_info.shape[0];

        const result = Physics.warpToKSpace(
            data,
            width, height,
            angleAxis, energyAxis,
            hv, workFunc
        );

        // Return new data and axes
        return {
            data: result.data,
            axes: {
                kx: Array.from(result.kAxis),
                energy: meta.axes.energy // Energy axis doesn't change in this step (warping is X-axis)
            },
            shape: [result.height, result.width]
        };
    }

    static convert3D(data, meta, hv, workFunc, innerPot, mappingMode, kzMode) {
        console.log("KConverter: Converting 3D to k-space...");

        // Axes: dim0=Energy, dim1=Angle, dim2=Scan(AngleY or hv)
        const axes = {
            dim0: meta.axes.energy,
            dim1: meta.axes.kx, // AngleX
            dim2: meta.axes.ky  // AngleY or hv
        };
        const shape = meta.data_info.shape; // [E, Ax, Ay/hv]

        let result;
        let newAxes = {};
        let units = {};

        if (mappingMode === 'kx-ky') {
            result = Physics.warp3D_KxKy(data, shape, axes, hv, workFunc);
            newAxes.kx = Array.from(result.axes.dim1);
            newAxes.ky = Array.from(result.axes.dim2);
            units.kx = "1/Å";
            units.ky = "1/Å";
        } else {
            // kx-kz
            const keepHv = kzMode === 'keep';
            result = Physics.warp3D_KxKz(data, shape, axes, workFunc, innerPot, keepHv);
            newAxes.kx = Array.from(result.axes.dim1);
            newAxes.ky = Array.from(result.axes.dim2); // This is now kz or hv
            units.kx = "1/Å";
            units.ky = keepHv ? "eV" : "1/Å";
        }

        return {
            data: result.data,
            axes: newAxes,
            shape: result.shape,
            units: units
        };
    }
}

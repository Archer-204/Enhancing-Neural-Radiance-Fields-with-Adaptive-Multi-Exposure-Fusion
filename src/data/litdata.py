# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Aleth-NeRF (https://github.com/cuiziteng/Aleth-NeRF)
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from typing import Optional

import gin

from src.data.data_util.blender import load_blender_data
from src.data.data_util.blender_ms import load_blender_ms_data
from src.data.data_util.lf import load_lf_data
from src.data.data_util.llff import load_llff_data
from src.data.data_util.nerf_360_v2 import load_nerf_360_v2_data
from src.data.data_util.refnerf_real import load_refnerf_real_data
from src.data.data_util.shiny_blender import load_shiny_blender_data
from src.data.data_util.tnt import load_tnt_data
from src.data.data_util.llff_single import load_image_data  # Single Image for LOL dataset
from src.data.data_util.llff_pair import load_llff_data_pair
from src.data.interface import LitData


## Load Single Image 
@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataSingle(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        # LLFF specific arguments
        llffhold: int = 8,
        spherify: bool = False,
        path_zflat: bool = False,
        offset: int = 250,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ):
        try:
            ndc_coord = gin.query_parameter("LitData.ndc_coord")
        except:
            ndc_coord = (-1.0, -1.0)
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_image_data(
            datadir=datadir,
            scene_name=scene_name,
            ndc_coord=ndc_coord,
            recenter=True,
            bd_factor=0.75,
            spherify=spherify,
            llffhold=llffhold,
            path_zflat=path_zflat,
            near=near,
            far=far,
        )
        self.scene_name = scene_name

        super(LitDataSingle, self).__init__(datadir)

        radx = 1 + 2 * offset / self.image_sizes[0][1]
        rady = 1 + 2 * offset / self.image_sizes[0][0]
        radz = 1.0
        self.scene_radius = [radx, rady, radz]
        self.use_sphere_bound = False

# LLFF dataset
@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataLLFF_Pair(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        # LLFF specific arguments
        factor: int = 8,    # Here use factor 8 for faster training
        llffhold: int = 8,
        spherify: bool = False,
        path_zflat: bool = False,
        offset: int = 250,
        # MipNeRF360 specific
        near: Optional[float] = None,
        far: Optional[float] = None,
    ):
        try:
            ndc_coord = gin.query_parameter("LitData.ndc_coord")
        except:
            ndc_coord = (-1.0, -1.0)
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_llff_data_pair(
            datadir=datadir,
            scene_name=scene_name,
            factor=factor,
            ndc_coord=ndc_coord,
            recenter=True,
            bd_factor=0.75,
            spherify=spherify,
            llffhold=llffhold,
            path_zflat=path_zflat,
            near=near,
            far=far,
        )

        super(LitDataLLFF_Pair, self).__init__(datadir)

        radx = 1 + 2 * offset / self.image_sizes[0][1]
        rady = 1 + 2 * offset / self.image_sizes[0][0]
        radz = 1.0
        self.scene_radius = [radx, rady, radz]
        self.use_sphere_bound = False

# LLFF dataset
@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataLLFF(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        # LLFF specific arguments
        factor: int = 8,    # Here use factor 8 for faster training
        llffhold: int = 8,
        spherify: bool = False,
        path_zflat: bool = False,
        offset: int = 250,
        # MipNeRF360 specific
        near: Optional[float] = None,
        far: Optional[float] = None,
    ):
        try:
            ndc_coord = gin.query_parameter("LitData.ndc_coord")
        except:
            ndc_coord = (-1.0, -1.0)
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_llff_data(
            datadir=datadir,
            scene_name=scene_name,
            factor=factor,
            ndc_coord=ndc_coord,
            recenter=True,
            bd_factor=0.75,
            spherify=spherify,
            llffhold=llffhold,
            path_zflat=path_zflat,
            near=near,
            far=far,
        )

        super(LitDataLLFF, self).__init__(datadir)

        radx = 1 + 2 * offset / self.image_sizes[0][1]
        rady = 1 + 2 * offset / self.image_sizes[0][0]
        radz = 1.0
        self.scene_radius = [radx, rady, radz]
        self.use_sphere_bound = False


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataBlender(LitData):
    def __init__(
        self,
        datadir: str,
        datadir_GAN: str,
        scene_name: str,
        # Compare method
        compare_method: str = "",
        # Blender specific
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
        cam_scale_factor: float = 1.0,
        white_bkgd: bool = True,
    ):

        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_blender_data(
            datadir=datadir,
            scene_name=scene_name,
            compare_method=compare_method,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            cam_scale_factor=cam_scale_factor,
            white_bkgd=white_bkgd,
        )

        (
            self.images2,
            self.intrinsics2,
            self.extrinsics2,
            self.image_sizes2,
            self.near2,
            self.far2,
            self.ndc_coeffs2,
            (self.i_train2, self.i_val2, self.i_test2, self.i_all2),
            self.render_poses2,
        ) = load_blender_data(
            datadir=datadir_GAN,
            scene_name=scene_name,
            compare_method=compare_method,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            cam_scale_factor=cam_scale_factor,
            white_bkgd=white_bkgd,
        )

        super(LitDataBlender, self).__init__(datadir)
        self.white_bkgd = white_bkgd


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataTnT(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        # TnT specific
        cam_scale_factor: float = 0.95,
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
        # MipNeRF360 specific
        near: Optional[float] = None,
        far: Optional[float] = None,
    ):
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_tnt_data(
            datadir=datadir,
            scene_name=scene_name,
            cam_scale_factor=cam_scale_factor,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            near=near,
            far=far,
        )

        super(LitDataTnT, self).__init__(datadir)


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataLF(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        # LF specific
        cam_scale_factor: float = 0.95,
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
        # MipNeRF360 specific
        near: Optional[float] = None,
        far: Optional[float] = None,
    ):
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_lf_data(
            datadir=datadir,
            scene_name=scene_name,
            cam_scale_factor=cam_scale_factor,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            near=near,
            far=far,
        )

        super(LitDataLF, self).__init__(datadir)


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataBlenderMultiScale(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        # Blender specific
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
        cam_scale_factor: float = 1.0,
        white_bkgd: bool = True,
    ):

        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
            self.multlosses,
        ) = load_blender_ms_data(
            datadir=datadir,
            scene_name=scene_name,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            cam_scale_factor=cam_scale_factor,
            white_bkgd=white_bkgd,
        )

        super(LitDataBlenderMultiScale, self).__init__(datadir)
        self.white_bkgd = white_bkgd


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataNeRF360V2(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        factor: int = 4,
        cam_scale_factor: float = 0.95,
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
        near: Optional[float] = None,
        far: Optional[float] = None,
        strict_scaling: bool = False,
    ):
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_nerf_360_v2_data(
            datadir=datadir,
            scene_name=scene_name,
            factor=factor,
            cam_scale_factor=cam_scale_factor,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            near=near,
            far=far,
            strict_scaling=strict_scaling,
        )

        super(LitDataNeRF360V2, self).__init__(datadir)


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataShinyBlender(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        # ShinyBlender specific
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
        cam_scale_factor: float = 1.0,
        white_bkgd: bool = True,
    ):

        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
            self.normals,
        ) = load_shiny_blender_data(
            datadir=datadir,
            scene_name=scene_name,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
            cam_scale_factor=cam_scale_factor,
            white_bkgd=white_bkgd,
        )

        super(LitDataShinyBlender, self).__init__(datadir)
        self.white_bkgd = white_bkgd


@gin.configurable(denylist=["datadir", "scene_name"])
class LitDataRefNeRFReal(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        factor: int = 4,
        cam_scale_factor: float = 0.95,
        train_skip: int = 1,
        val_skip: int = 1,
        test_skip: int = 1,
    ):
        (
            self.images,
            self.intrinsics,
            self.extrinsics,
            self.image_sizes,
            self.near,
            self.far,
            self.ndc_coeffs,
            (self.i_train, self.i_val, self.i_test, self.i_all),
            self.render_poses,
        ) = load_refnerf_real_data(
            datadir=datadir,
            scene_name=scene_name,
            factor=factor,
            cam_scale_factor=cam_scale_factor,
            train_skip=train_skip,
            val_skip=val_skip,
            test_skip=test_skip,
        )

        super(LitDataNeRF360V2, self).__init__(datadir)

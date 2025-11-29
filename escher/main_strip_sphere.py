# Main Strip

# Libraries
import os

# INIT
from escher.geometry.get_base_mesh import get_2d_square_mesh, get_hexagonal_mesh
import escher.guidance.sd as sd
from rich.pretty import pprint
import torch
import numpy as np
from escher.OTE.core import OTESolver, SOTESolver
import escher.geometry.split_square_boundary as split_square_boundary
import igl
from omegaconf import OmegaConf
import h5py
from scipy.io import loadmat



# get path to this file
from pathlib import Path
PATH = Path(__file__).parent.absolute()

class Escher:
    def __init__(self) -> None:
        # Load base.yaml file 
        # args = parser.parse_args()
        cli_conf = OmegaConf.from_cli()
        conf_file = cli_conf.get("CONF_FILE", "configs/base.yaml")
        base_conf = OmegaConf.load(PATH / f"{conf_file}")
        args = OmegaConf.merge(base_conf, cli_conf)
        self.args = args
        self.device = torch.device(args.DEVICE)

        self.testLBFGS()


        self.init_guidance()
        print("Guidance model loaded")
            # self.init_backgrounds()
            # print("Backgrounds loaded")
        self.get_embeddings()
        print("Embeddings loaded")

        self.init_mesh_and_solver()
        print("Mesh and solver loaded")
        # self.init_optimizer()
        # print("Optimizer loaded")
        # self.init_loss()
        # print("Loss loaded")
        # if self.args.SYMMETRY_EXPERIMENT:
        #     self.init_symmetry_experiment()

    def init_guidance(self):
        # ================== Init Stable Diffusion ===========================
        textual_inversion_path = ""
        if self.args.TEXTUAL_INVERSION:
            textual_inversion_path = "/home/groueix/GenerativeEscherPatterns/diffusers/examples/textual_inversion/textual_inversion_cat/learned_embeds.safetensors"

        config = sd.Config(
            pretrained_model_name_or_path = self.args.PRETRAINED_MODEL_NAME_OR_PATH,
            textual_inversion=textual_inversion_path,
            guidance_scale=self.args.GUIDANCE_SCALE,
            half_precision_weights=self.args.USE_HALF_PRECISION,
            grad_clip=[0, 2.0, 8.0, 1000] if self.args.CLIP_GRADIENTS_IN_SDS else None,
        )
        pprint(config)
        self.guidance_model = sd.StableDiffusion(config)

    def get_embeddings(self):
        if isinstance(self.args.PROMPT, str):
            self.args.PROMPT = [self.args.PROMPT]

        with torch.no_grad():
            negative_embedding = self.guidance_model.get_text_embeds(self.args.NEGATIVE_PROMPT)
            prompt_embedding = [self.guidance_model.get_text_embeds(prompt) for prompt in self.args.PROMPT]
            n_prompt = len(self.args.PROMPT)

            if n_prompt == 1:
                self.text_embeds = torch.cat(
                    prompt_embedding * self.args.IMAGE_BATCH_SIZE + [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                )
            elif n_prompt == 2:
                # The text batch is organized as follows
                # [prompt1, ..., prompt1 |  prompt2, ..., prompt2 |  negative, ..., negative |  negative, ..., negative]
                # the corresponding rendering batch is organized as follows
                # [mesh1, ..., mesh1 |  mesh2, ..., mesh2]
                assert self.args.IMAGE_BATCH_SIZE % 2 == 0, "Batch size must be even"
                batch_size_per_prompt = self.args.IMAGE_BATCH_SIZE // 2
                positive_embedding = [prompt_embedding[0]] * batch_size_per_prompt + [
                    prompt_embedding[1]
                ] * batch_size_per_prompt
                negative_embedding = [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                self.text_embeds = torch.cat(positive_embedding + negative_embedding)
                self.batch_size_per_prompt = batch_size_per_prompt
            else:
                assert (np.sqrt(n_prompt) % 1 == 0), "Number of prompts must be a square number"
                # The text batch is organized as follows
                # [prompt1, ..., prompt1 |  prompt2, ..., prompt2 | ... | prompt_k,..., prompt_k | negative, ..., negative |  negative, ..., negative |  negative, ..., negative]
                # the corresponding rendering batch is organized as follows
                # [mesh1, ..., mesh1 |  mesh2, ..., mesh2 | ... |  meshk, ..., meshk]
                assert self.args.IMAGE_BATCH_SIZE % n_prompt == 0, "Batch size must be a multiple of n_prompt"
                batch_size_per_prompt = self.args.IMAGE_BATCH_SIZE // n_prompt
                positive_embedding = []
                for i in range(n_prompt):
                    positive_embedding += [prompt_embedding[i]] * batch_size_per_prompt
                negative_embedding = [negative_embedding] * self.args.IMAGE_BATCH_SIZE
                self.text_embeds = torch.cat(positive_embedding + negative_embedding)
                self.batch_size_per_prompt = batch_size_per_prompt

            del self.guidance_model.text_encoder

    def init_mesh_and_solver(self):
        # ============== generate a 2D mesh of a square =======================
        assert self.args.MESH_RESOLUTION % 2 == 0, "mesh resolution must be even for some wallpaper groups"
        print(len(self.args.PROMPT))
        assert (np.sqrt(len(self.args.PROMPT)) % 1 == 0) or len(
            self.args.PROMPT
        ) == 2, "number of prompts must be a square number, or 2"

        points, faces_npy, faces_split, mask = get_2d_square_mesh(
            self.args.MESH_RESOLUTION, num_labels=len(self.args.PROMPT)
        )

        print("POINTS", len(points), points[0])
        print("FACES NPY", len(faces_npy), faces_npy[0])
        print("FACES SPLIT", len(faces_split), faces_split[0][0])

        # =========== init uv ==================================================
        normalized_points = points
        normalized_points = normalized_points - normalized_points.min()
        uv = normalized_points / normalized_points.max() # [0,1]
        uv = torch.from_numpy(uv).unsqueeze(0).to(self.device)
        normalized_points = 2 * normalized_points / normalized_points.max() - 1 # [-1,1]

        # tri = Delaunay(points)
        faces = torch.from_numpy(faces_npy)

        # bdry indices of mesh
        bdry = igl.boundary_loop(faces_npy)

        # split the bdry into 4 sides (left,right,top,down)
        if not ("Hexagon" in self.args.TILING_TYPE):
            sides = split_square_boundary.split_square_boundary(points, bdry)

        # generate nx2 list of edge pairs (i,j)
        adjacency_list = igl.adjacency_list(faces_npy)  # list of lists containing at index i the adjacent vertices of vertex i

        edge_pairs = []
        for r, i in zip(adjacency_list, range(len(adjacency_list))):
            for j in r:
                if i < j:
                    edge_pairs.append((i, j))
        edge_pairs = np.asarray(edge_pairs)
        print(f'EDGE PAIRS SHAPE: {edge_pairs.shape}')


        constraint_data = self.constraints_from_args(points, sides)

        # the solver itself
        solver = OTESolver.OTESolver(edge_pairs, points, constraint_data)
        # specify trainable parameter in pytorch -> in this case we want to specify theta
        # a scalar multiple of the edges (i,j)
        W = torch.nn.Parameter(torch.randn((edge_pairs.shape[0], 1)))
        self.faces_split = [
            torch.from_numpy(faces_split_).to(self.device).type(torch.int32) for faces_split_ in faces_split
        ]

        self.points = points
        # UV referring to the coordinate system
        self.uv = uv
        self.bdry = bdry
        self.faces = faces
        self.faces_npy = faces_npy
        self.constraint_data = constraint_data
        # self.ROTATION_MATRIX = ROTATION_MATRIX
        # self.global_map = GlobalDeformation.GlobalDeformation(
        #     constraint_data.get_horizontal_symmetry_orientation(), device=self.device
        # )
        # self.global_map.to(self.device)
        self.solver = solver
        self.W = W
        self.sides = sides
        self.edge_pairs = edge_pairs

    def run(self):
        # MAIN LOOP 
        for iter in range(self.args.N_STEPS):

            # No Texture Loop
            if iter < self.args.ONLY_TEXTURE_FROM_THIS_POINT:
                if self.args.CLAMP_TEXTURE:
                    self.color_parameters.data = self.color_parameters.data.clip(0, 1)

                self.optimizer.zero_grad()
                # weights are positive and smaller than 1
                if self.args.SIGMOID_WEIGHTS:
                    # self.W.data = self.W.data.clip(-10, 10) #after 10 we're praxtically at 1 for sigmoid This is killing gradients
                    w = torch.special.expit(self.W)
                else:
                    self.W.data = self.W.data.clip(0, 1)
                    w = self.W
                w_solver_input = w * self.args.W_RANGE + (1 - self.args.W_RANGE) / 2
                print(w_solver_input)
                # [0,1] ----> [r, 1-r] where r = (1-W_RANGE)/2

                # ======Solve linear solve ==========================================================
                mapped, _, success = self.solver.solve(w_solver_input)
                mapped = mapped.cuda().float()
                print(mapped.shape)
            
            # Create a Visualization here - 

    def testLBFGS(self):
        print(os.getcwd())
        # import test matrices for SOTE Solver
        N = loadmat('N.mat')['N'].toarray()
        x0 = loadmat('x0.mat')['x0']
        w = loadmat('wMat.mat')['wmat'].toarray()
        A = loadmat('A.mat')['A'].toarray()
        b = loadmat('b.mat')['b']
        print(b.shape)
        L = loadmat('L.mat')['L'].toarray()
        V = loadmat('V.mat')['V']
        F = loadmat('F.mat')['F']
        
        solver = SOTESolver.SOTESolver(V, F, x0, N, w, A, L)
        x = solver.solve()
        

if __name__ == "__main__":
    escher = Escher()
    escher.testLBFGS()
    exit()
    if escher.args.EXPERIMENT == "OUTLINE":
        escher.outline()
    elif escher.args.EXPERIMENT == "IMAGE_LOSS":
        escher.run()
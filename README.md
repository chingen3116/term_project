# term_project
* MiDaS.py : only midas model.
* Version 2 : including DIP_term_project.py and combined_image_V2.jpg
  - estimate_depth_map
  - calculate_edge_density
  - combine_maps
  - apply_variable_blur
* Version 3 : including V3.py and combined_image_V3.jpg
  - compute_foregroung_mask
  - compute_focus_map_from_mask
  - apply_strong_blur
* Version 4 : including V4.py and combined_image_V4_C.jpg and combined_image_V4_M.jpg
  - Refine compute_foreground_mask
    - Visualize the mask after each morphological operation (binary_edges.jpg, dilated.jpg, dilated.jpg)
    - Remove shape filters (aspect_ratio)
    - Set min_area to 1% of the total pixels of the image
  - Modify the parameters of Canny 

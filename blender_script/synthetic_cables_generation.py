import bpy
import numpy as np
from numpy import random
import time
import re
import math
import os
import cv2


class ModalTimerOperator(bpy.types.Operator):
    #Operator which runs itself from a timer
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    #Variation limits. Dummy values, adjust depending on dimensions and mass of your cables
    cables_list = ['cable1', 'cable2', 'cable3', 'cable4', 'cable5', 'cable6', 'cable7', 'cable8'] #Maximum number of cables in the scene
    camera_distance = [0.09, 0.12, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.27, 0.30, 0.33] # In m
    limits_vertex_mass = [0.3, 0.5] #Cable deformation
    limits_position_global = {'x':25,'y':35} #In mm. In both directions
    limits_position_aligned = {'x':12,'y':14} #In mm. In both directions
    maximum_angle_aligned = 10 #In deg. In both directions
    max_wind_strength = 10 #Dummy value

    #Initialize variables
    adjust_textures_path = True
    all_node_attributes = ["COLOR","AO","NORMAL","BUMP","GLOSS","DISPLACEMENT","METALNESS","OVERLAY","ALPHA"]
    _timer = None
    count = 0
    img_number = 0
    n_cables_out = 0
    cable_colors = []
    img_name = ""    
    floor_materials = []
    aligned_opt = True
    close_bkg_opt = True
    HDR_light_opt = False

    #Define how many images to create from each category
    images_type = {'close': {'n': 3000, 'aligned': False, 'close_bkg': True, 'HDR_light': False, 'dir': 'train_imgs_close\\'},
                    'close_aligned': {'n': 3000, 'aligned': True, 'close_bkg': True, 'HDR_light': False, 'dir': 'train_imgs_close_aligned\\'},
                    'HDR_close': {'n': 3000, 'aligned': False, 'close_bkg': True, 'HDR_light': True, 'dir': 'train_imgs_close_HDR\\'},
                    'HDR_close_aligned': {'n': 3000, 'aligned': True, 'close_bkg': True, 'HDR_light': True, 'dir': 'train_imgs_close_aligned_HDR\\'},
                    'HDR_far': {'n': 3000, 'aligned': False, 'close_bkg': False, 'HDR_light': True, 'dir': 'train_imgs_far\\'},
                    'HDR_far_aligned': {'n': 3000, 'aligned': True, 'close_bkg': False, 'HDR_light': True, 'dir': 'train_imgs_far_aligned\\'}}
    types_added = []

    #Define paths
    imgs_path = ""
    current_path = os.path.dirname(os.path.realpath(__file__))+"\\..\\"
    general_path = current_path+"\\Dataset\\"
    textures_path = current_path+"\\Texture\\"
    hdr_path = current_path+"\\HDR"
    dark_hdr = ["colorful_studio_4k.hdr", "gear_store_4k.hdr", "storeroom_4k.hdr", "christmas_photo_studio_04_4k.hdr"]
    path_print = current_path+"\\info.txt"


    def modal(self, context, event):
        """
        Manager method: Checks the animation status and calls the cancel method when the maximum frame is reached
        """
        if bpy.context.scene.frame_current > 100:
            self.cancel(context) #When frame 100 is reached, it stops the animation and renders the images
        if event.type == 'TIMER':
            self.count += 1
        return {'PASS_THROUGH'}


    def execute(self, context):
        wm = context.window_manager
        #Get possible HDRs
        self.hdr_names = os.listdir(self.hdr_path)
        #Get possible floor materials
        materials_obj = bpy.data.objects['All_materials']
        for material in materials_obj.material_slots:
            mat_name = str(re.findall('"([^"]*)"', str(material))[0])
            self.floor_materials.append(mat_name)
        if self.adjust_textures_path:
            self.adjust_textures()
            self.adjust_textures_path = False
        #Configure option parameters based on the iamge category
        end = True
        for type in self.images_type:
            if not (type in self.types_added):
                end = False
                self.types_added.append(type)
            else:
                continue
            self.total_img_number = self.images_type[type]['n']
            self.imgs_path = str(self.general_path) + self.images_type[type]['dir']
            self.aligned_opt = self.images_type[type]['aligned']
            self.close_bkg_opt = self.images_type[type]['close_bkg']
            self.HDR_light_opt = self.images_type[type]['HDR_light']
            if self.images_type[type]['n']>0:
                break
        if end:
            return {'CANCELLED'}
        #Apply random modifications
        self.cables_modifications(context)
        #Start animating
        bpy.context.scene.frame_current = 0
        bpy.ops.screen.animation_play()
        self._timer = wm.event_timer_add(1, window=context.window)
        self.count = 0
        wm.modal_handler_add(self) #Adds modal_handler
        return {'RUNNING_MODAL'}


    def adjust_textures(self):
        """
        Loads all the textures and defines the floor materials
        """
        for material in self.floor_materials:
            texture_nodes = bpy.data.materials[material].node_tree.nodes[material].node_tree.nodes
            for node_attrib in self.all_node_attributes:
                if node_attrib in texture_nodes:
                    texture_attrib_img_name = (str(texture_nodes[node_attrib].image).split('"')[1]).split('.')[0]
                    extension_img = texture_nodes[node_attrib].image.filepath.split('.')[-1]
                    new_texture_attrib_path = self.textures_path + texture_attrib_img_name.split('_')[0] + '\\' + texture_attrib_img_name + "." + extension_img
                    new_image = bpy.data.images.load(new_texture_attrib_path)
                    bpy.data.materials[material].node_tree.nodes[material].node_tree.nodes[node_attrib].image = new_image
                
    
    def cancel(self, context):
        """
        Frame 100 has been reached: Stops the animation and renders images
        """
        bpy.ops.screen.animation_cancel(restore_frame=False)

        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        
        bpy.context.scene.frame_set(100)
        self.img_name = str(round(time.time()))
        bpy.context.scene.render.filepath = str(self.imgs_path)+self.img_name+'_image.png'
        bpy.ops.render.render(write_still = True) #Render image
        
        self.render_filter(context) #Render segmentation mask
        
        #Repeats the process for a new image if there are still images to be created
        if self.img_number < self.total_img_number:
            self.repeat_process(context)
        else:
            #Configure option parameters
            end = True
            for type in self.images_type:
                if not (type in self.types_added):
                    end = False
                    self.types_added.append(type)
                else:
                    continue
                self.total_img_number = self.images_type[type]['n']
                self.imgs_path = str(self.general_path) + self.images_type[type]['dir']
                self.aligned_opt = self.images_type[type]['aligned']
                self.close_bkg_opt = self.images_type[type]['close_bkg']
                self.HDR_light_opt = self.images_type[type]['HDR_light']
                if self.images_type[type]['n']>0:
                    break
            if end:
                return {'CANCELLED'}
            else:
                self.img_number = 0
                self.repeat_process(context)

        
    def render_filter(self, context):
        """
        Renders the segmentation mask: Changes cables material to white color and floor material to black color, renders the image, binarizes it, and saves it
        """
        white_mat = bpy.data.materials.get("white_cable_material")
        for cable_name in self.cables_list:
            #Change material
            cable_obj = bpy.data.objects[str(cable_name)]
            cable_obj.data.materials[0] = white_mat
            
        #Floor material change
        floor_obj = bpy.data.objects['Floor']
        floor_obj.hide_render = False
        # Get material
        mat = bpy.data.materials.get("floor_material")
        if mat is None:
            # create material
            mat = bpy.data.materials.new(name="floor_material")
        # Assign it to object
        if floor_obj.data.materials:
            # assign to 1st material slot
            floor_obj.data.materials[0] = mat
        else:
            # no slots
            floor_obj.data.materials.append(mat)
        
        bpy.context.scene.render.filepath = str(self.imgs_path)+self.img_name+'_filter.png'
        bpy.ops.render.render(write_still = True)
        filter_img1 = cv2.imread(str(self.imgs_path)+self.img_name+'_filter.png', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(filter_img1, 60, 255, cv2.THRESH_BINARY)
        rgb_filter_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB) # Convert grayscale to RGB
        cv2.imwrite(str(self.imgs_path)+self.img_name+'_filter.png', rgb_filter_image) 
        self.cable_colors = []
         
            
    def repeat_process(self, context):
        """
        Applies random modifications and repeats the process to create a new image
        """
        wm = context.window_manager
        bpy.context.scene.frame_current = 0
        self.cables_modifications(context) #Apply random modifications
        bpy.ops.screen.animation_play()
        self._timer = wm.event_timer_add(1, window=context.window)
        self.count = 0
        self.img_number += 1
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    
    def cables_modifications(self, context):
        """
        Applies random modifications to the cables, floor, HDRI, wind, lights, and camera
        """
        height = 1
        
        #CHANGE FLOOR
        new_floor_material = self.floor_materials[random.randint(0,len(self.floor_materials))]
        floor_obj = bpy.data.objects['Floor']
        if self.close_bkg_opt:
            floor_obj.hide_render = False
            # Get material
            mat = bpy.data.materials.get(new_floor_material)
            floor_obj.data.materials[0] = mat
            #Change material pattern (x, y coord, angle)
            bpy.data.materials[new_floor_material].node_tree.nodes[new_floor_material].inputs[3].default_value = random.randint(-100,100)
            bpy.data.materials[new_floor_material].node_tree.nodes[new_floor_material].inputs[4].default_value = random.randint(-100,100)
            bpy.data.materials[new_floor_material].node_tree.nodes[new_floor_material].inputs[5].default_value = random.randint(0,360) #deg
            bpy.data.materials[new_floor_material].node_tree.nodes[new_floor_material].inputs[1].default_value = (float(random.randint(7,15))/10) #scale
        else:
            floor_obj.hide_render = True
            
        #CHANGE WIND
        wind_obj = bpy.data.objects["Wind"]
        wind_deg = float(random.randint(0, 360)) #Wins position: degrees
        wind_deg = (wind_deg*math.pi)/180
        wind_R = float(random.randint(50, 80))/10 #Wind position: radius
        wind_obj.location[0] = wind_R * math.sin(wind_deg)
        wind_obj.location[1] = wind_R * math.cos(wind_deg)
        wind_obj.location[2] = float(random.randint(0, 30))/10
        wind_obj.rotation_euler = ((float(random.randint(60, 120))*math.pi)/180,0,-wind_deg)
        wind_obj.field.strength = (random.randint(0, self.max_wind_strength)) 
        wind_obj.field.flow = wind_obj.field.strength/5
        
        #CHANGE CABLES
        self.n_cables_out = random.randint(0,len(self.cables_list)-1)
        index = 0
        cable_rot_z = float(random.randint(0, 360))
        if self.aligned_opt:
            cable_mass = float(random.randint(int(self.limits_vertex_mass[0]*10), int(self.limits_vertex_mass[1]*10)))/10
        else:
            cable_mass = float(random.randint(int(self.limits_vertex_mass[0]*10), int(self.limits_vertex_mass[1]*10)))/10
        for cable_name in self.cables_list: #Modifications for each cable
            if index < (len(self.cables_list)-self.n_cables_out):
                bpy.data.objects[cable_name].hide_render = False
                self.modify_cable(context, cable_name, height, cable_rot_z, cable_mass)
            else: #Out of the scene
                bpy.data.objects[cable_name].hide_render = True
                self.modify_cable(context, cable_name, height, cable_rot_z, cable_mass, out_scene = True)
            index += 1
            height += 1
            
        #CHANGE LIGHT
        if not self.HDR_light_opt:
            bpy.data.worlds["World"].node_tree.nodes["Mix Shader"].inputs[0].default_value = 1 #Not HDR
            light_type = random.randint(1, 2) #1/2 point light, 1/2 bar lights
            if light_type != 2:
                light_obj = bpy.data.objects["Light"]
                area_light_obj = bpy.data.objects["Area"]
                light_obj.hide_render = False
                area_light_obj.hide_render = True
                light_deg = float(random.randint(0, 360)) #Light position: degrees
                light_deg = (light_deg*math.pi)/180
                light_R = float(random.randint(50, 65))/10 #Light position: radius
                light_obj.location[0] = light_R * math.cos(light_deg)
                light_obj.location[1] = light_R * math.sin(light_deg)
                light_obj.location[2] = float(random.randint(50, 65))/10
                light_obj.data.energy = (random.randint(800, 1500)) #Power
                light_obj.data.shadow_soft_size = float(random.randint(1, 20))/10 #Radius (how much shadow)
            else:
                light_obj = bpy.data.objects["Light"]
                area_light_obj = bpy.data.objects["Area"]
                light_obj.hide_render = True
                area_light_obj.hide_render = False
                possible_loc = [float(random.randint(-150, 40))/10, float(random.randint(40, 150))/10]
                area_light_obj.location[0] = random.choice(possible_loc)
                area_light_obj.data.energy = (random.randint(400, 1800))
            
        #CHANGE LIGHTS HDR
        else:
            bpy.data.worlds["World"].node_tree.nodes["Mix Shader"].inputs[0].default_value = 0 #HDR
            light_obj = bpy.data.objects["Light"]
            area_light_obj = bpy.data.objects["Area"]
            light_obj.hide_render = True
            area_light_obj.hide_render = True
            if self.close_bkg_opt:
                hdr_name = "machine_shop_01_4k.hdr"
                new_hdr_path = self.hdr_path + "//" + hdr_name
                new_hdr = bpy.data.images.load(new_hdr_path)
                bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image = new_hdr
                x_rot_HDR = [0, math.pi/2]
                bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[0].default_value[2] = random.choice(x_rot_HDR)
                #Rotate in Z
                rot_deg = float(random.randint(-180, 180)) #Light position: degrees
                rot_deg = (rot_deg*math.pi)/180
                bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[2] = rot_deg
            else:
                hdr_name = str(random.choice(self.hdr_names))
                new_hdr_path = self.hdr_path + "//" + hdr_name
                new_hdr = bpy.data.images.load(new_hdr_path)
                bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image = new_hdr
                #Rotate in Z
                rot_deg = float(random.randint(-180, 180)) #Light position: degrees
                rot_deg = (rot_deg*math.pi)/180
                bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[2] = rot_deg
                #Change Brightness
                if hdr_name in self.dark_hdr:
                    bpy.data.worlds["World"].node_tree.nodes["Mix Shader.001"].inputs[0].default_value = float(random.randint(10, 60))/100
                else:
                    bpy.data.worlds["World"].node_tree.nodes["Mix Shader.001"].inputs[0].default_value = 0
        
        #CHANGE CAMERA POSITION (CABLE SIZE)
        camera_obj = bpy.data.objects["Camera"]
        self.chosen_distance = random.choice(self.camera_distance)
        camera_obj.location[2] = self.chosen_distance
        camera_obj.data.dof.focus_distance = camera_obj.location[2]        
                
    
    def modify_cable(self, context, cable_name, height, cable_rot, cable_mass, out_scene = False):
        """
        Applies random modifications to one cable
        """
        cable_obj = bpy.data.objects[cable_name]
        
        #CHANGE PHYSICAL PROPERTIES
        cable_obj.modifiers["Cloth"].settings.mass = cable_mass
        
        #CHANGE POSITION
        if out_scene:
            cable_obj.location = (60.0,60.0,1.0) #Not interferring with other cables
        elif self.aligned_opt:
            cable_obj.location = (float(random.randint(-self.limits_position_aligned['x'], self.limits_position_aligned['x'])/1000),float(random.randint(-self.limits_position_aligned['y'], self.limits_position_aligned['y'])/1000),height) #If we want less cables in the image, just move them outside the camera view
        else:
            cable_obj.location = (float(random.randint(-self.limits_position_global['x'], self.limits_position_global['x'])/1000),float(random.randint(-self.limits_position_global['y'], self.limits_position_global['y'])/1000),height)
        if self.aligned_opt:
            cable_rot_z = ((cable_rot + float(random.randint(-self.maximum_angle_aligned, self.maximum_angle_aligned)))*math.pi)/180
            cable_obj.rotation_euler = (0,0,cable_rot_z)
        else:
            cable_obj.rotation_euler = (0,0,float(random.randint(0, 314)/100))
    
        #CHANGE CABLE MATERIAL
        mat_name = str(cable_name)+"_material"
        mat = bpy.data.materials.get(mat_name)
        cable_obj.data.materials[0] = mat
        #Change color
        color_rgb = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]
        self.cable_colors.append(color_rgb)
        bpy.data.materials[mat_name].node_tree.nodes["ColorRamp"].color_ramp.elements[0].color = (color_rgb[0]/255, color_rgb[1]/255, color_rgb[2]/255, 1)
        #Dirt density and pattern
        bpy.data.materials[mat_name].node_tree.nodes["ColorRamp"].color_ramp.elements[0].position = random.randint(65, 89)/100
        bpy.data.materials[mat_name].node_tree.nodes["Mapping"].inputs[0].default_value[1] = random.randint(-500, 500)
        bpy.data.materials[mat_name].node_tree.nodes["Mapping"].inputs[1].default_value[1] = random.randint(-500, 500)
        #Dirt color (Dark)
        color_dirt = [random.randint(0, 4),random.randint(0, 4),random.randint(0, 4)]
        bpy.data.materials[mat_name].node_tree.nodes["ColorRamp"].color_ramp.elements[1].color = (color_dirt[0]/255, color_dirt[1]/255, color_dirt[2]/255, 1)
        #Roughness and metalic
        bpy.data.materials[mat_name].node_tree.nodes["ColorRamp.001"].color_ramp.elements[1].position = random.randint(60, 70)/100


def register():
    bpy.utils.register_class(ModalTimerOperator)


def unregister():
    bpy.utils.unregister_class(ModalTimerOperator)


if __name__ == "__main__":
    register()
    bpy.ops.wm.modal_timer_operator()    
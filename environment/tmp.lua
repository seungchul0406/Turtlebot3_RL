function sysCall_init() 
    -- First time we execute this script. 

    -- Make sure we have version 2.4.12 or above (the omni-wheels are not supported otherwise)
    v=sim.getInt32Parameter(sim.intparam_program_version)
    if (v<20412) then
        sim.displayDialog('Warning','The YouBot model is only fully supported from V-REP version 2.4.12 and above.&&nThis simulation will not run as expected!',sim.dlgstyle_ok,false,'',nil,{0.8,0,0,0,0,0})
    end

    --Prepare initial values and retrieve handles:
    wheelJoints={-1,-1,-1,-1} -- front left, rear left, rear right, front right
    wheelJoints[1]=vrep.simxGetObjectHandle('rollingJoint_fl')
    wheelJoints[2]=vrep.simxGetObjectHandle('rollingJoint_rl')
    wheelJoints[3]=vrep.simxGetObjectHandle('rollingJoint_rr')
    wheelJoints[4]=vrep.simxGetObjectHandle('rollingJoint_fr')
    youBot=sim.getObjectHandle('youBot')
    youBotRef=sim.getObjectHandle('youBot_ref')
    tip=sim.getObjectHandle('youBot_positionTip')
    target=sim.getObjectHandle('youBot_positionTarget')
    armJoints={-1,-1,-1,-1,-1}
    for i=0,4,1 do
        armJoints[i+1]=sim.getObjectHandle('youBotArmJoint'..i)
    end
    ui=simGetUIHandle('youBot_UI')
    simSetUIButtonLabel(ui,0,sim.getObjectName(youBot)..' user interface') -- Set the UI title (with the name of the current robot)
    ik1=sim.getIkGroupHandle('youBotUndamped_group')
    ik2=sim.getIkGroupHandle('youBotDamped_group')
    ikFailedReportHandle=-1
    forwBackVelRange={-240*math.pi/180,240*math.pi/180}  -- min and max wheel rotation vel. for backward/forward movement
    leftRightVelRange={-240*math.pi/180,240*math.pi/180} -- min and max wheel rotation vel. for left/right movement
    rotVelRange={-240*math.pi/180,240*math.pi/180}       -- min and max wheel rotation vel. for left/right rotation movement

    forwBackVel=0
    leftRightVel=0
    rotVel=0
    initSizeFactor=sim.getObjectSizeFactor(youBot) -- only needed if we scale the robot up/down
    
    -- desired joint positions, and desired cartesian positions:
    desiredJ={0,30.91*math.pi/180,52.42*math.pi/180,72.68*math.pi/180,0} -- when in FK mode
    for i=1,5,1 do
        sim.setJointPosition(armJoints[i],desiredJ[i])
    end
    desiredPos={0,0,0} -- when in IK mode
    currentPos={0,0,0} -- when in IK mode
    ikMinPos={-0.5*initSizeFactor,-0.2*initSizeFactor,-0.3*initSizeFactor}
    ikRange={1*initSizeFactor,1*initSizeFactor,0.9*initSizeFactor}

    -- We compute the initial position and orientation of the tip RELATIVE to the robot base (because the base is moving)
    initialTipPosRelative=sim.getObjectPosition(tip,youBotRef)--youBot)
    ikMode=false -- We start in FK mode
    maxJointVelocity=40*math.pi/180 
    maxPosVelocity=0.1*initSizeFactor
    previousS=initSizeFactor

    gripperCommunicationTube=sim.tubeOpen(0,'youBotGripperState'..sim.getNameSuffix(nil),1)
end
-- This example script is non-threaded (executed at each simulation pass)
-- The functionality of this script (or parts of it) could be implemented
-- in an extension module (plugin) and be hidden. The extension module could
-- also allow connecting to and controlling the real robot.


function sysCall_cleanup() 
 
end 

function sysCall_actuation() 
    -- s will scale a few values hereafter (has only an effect if the robot is scaled down/up)
    s=sim.getObjectSizeFactor(youBot) 
    if (s~=previousS) then
        f=s/previousS
        for i=1,3,1 do
            desiredPos[i]=desiredPos[i]*f
            currentPos[i]=currentPos[i]*f
            ikMinPos[i]=ikMinPos[i]*f
            ikRange[i]=ikRange[i]*f
            initialTipPosRelative[i]=initialTipPosRelative[i]*f
        end
        maxPosVelocity=maxPosVelocity*f
        previousS=s
    end
    
    buttonID=simGetUIEventButton(ui)
    if (buttonID==200) then -- Forward/backward slider was changed
        forwBackVel=forwBackVelRange[1]+simGetUISlider(ui,buttonID)*0.001*(forwBackVelRange[2]-forwBackVelRange[1])
    end
    if (buttonID==201) then -- left/right slider was changed
        leftRightVel=leftRightVelRange[1]+simGetUISlider(ui,buttonID)*0.001*(leftRightVelRange[2]-leftRightVelRange[1])
    end
    if (buttonID==202) then -- left/right rotation slider was changed
        rotVel=rotVelRange[1]+simGetUISlider(ui,buttonID)*0.001*(rotVelRange[2]-rotVelRange[1])
    end
    if (buttonID==212) then -- stop button was clicked
        forwBackVel=0
        leftRightVel=0
        rotVel=0
        -- Reset the wheel movement sliders to the neutral position:
        simSetUISlider(ui,200,500)
        simSetUISlider(ui,201,500)
        simSetUISlider(ui,202,500)
    end
    if (buttonID==211) then -- the open/close button was pressed:
        if (sim.boolAnd32(simGetUIButtonProperty(ui,buttonID),sim.buttonproperty_isdown)~=0) then
            sim.tubeWrite(gripperCommunicationTube,sim.packInt32Table({0})) -- close the gripper
        else
            sim.tubeWrite(gripperCommunicationTube,sim.packInt32Table({1})) -- open the gripper
        end
    end
    
    if (buttonID>=203)and(buttonID<=207) then -- we want to control the arm in FK mode!
        cyclic,interval=sim.getJointInterval(armJoints[buttonID-202])
        desiredJ[buttonID-202]=interval[1]+simGetUISlider(ui,buttonID)*0.001*interval[2]
        ikMode=false
    end
    
    if ((buttonID>=208)and(buttonID<=210)) then -- we want to control the arm in IK mode!
        desiredPos[buttonID-207]=ikMinPos[buttonID-207]+ikRange[buttonID-207]*simGetUISlider(ui,buttonID)/1000
        ikMode=true
    end
    
    if ikMode then
        -- We are in IK mode
        maxVariationAllowed=maxPosVelocity*sim.getSimulationTimeStep()
        deltaX={0,0,0}
        -- position:
        for i=1,3,1 do
            delta=desiredPos[i]-currentPos[i]
            if (math.abs(delta)>maxVariationAllowed) then
                delta=maxVariationAllowed*delta/math.abs(delta) -- we limit the variation to the maximum allowed
            end
            deltaX[i]=delta
        end
    
        currentPos={currentPos[1]+deltaX[1],currentPos[2]+deltaX[2],currentPos[3]+deltaX[3]}
    
        pos={initialTipPosRelative[1]+currentPos[1],initialTipPosRelative[2]+currentPos[2],initialTipPosRelative[3]+currentPos[3]}
        -- We set the desired position and orientation
        sim.setObjectPosition(target,youBotRef,pos)--youBot,pos)
    
        if (sim.handleIkGroup(ik1)==sim.ikresult_fail) then
            -- the position could not be reached.
            sim.handleIkGroup(ik2) -- Apply a damped resolution method
            if (ikFailedReportHandle==-1) then -- We display a IK failure (in pos) report message
                ikFailedReportHandle=sim.displayDialog("IK failure report","IK solver failed.",sim.dlgstyle_message,false,"",nil,{1,0.7,0,0,0,0})
            end
        else
            if (ikFailedReportHandle>=0) then
                sim.endDialog(ikFailedReportHandle) -- We close any report message about IK failure in orientaion
                ikFailedReportHandle=-1
            end
        end
        -- Now update the desiredJ in case we switch back to FK mode:
        for i=1,5,1 do
            desiredJ[i]=sim.getJointPosition(armJoints[i])
        end
        
    else
        -- We are in FK mode
        currentJ={0,0,0,0,0}
        for i=1,5,1 do
            currentJ[i]=sim.getJointPosition(armJoints[i])
        end
        maxVariationAllowed=maxJointVelocity*sim.getSimulationTimeStep()
        for i=1,5,1 do
            delta=desiredJ[i]-currentJ[i]
            if (math.abs(delta)>maxVariationAllowed) then
                delta=maxVariationAllowed*delta/math.abs(delta) -- we limit the variation to the maximum allowed
            end
            sim.setJointPosition(armJoints[i],currentJ[i]+delta)
        end
        -- Now make sure that everything is ok if we switch to IK mode:
        sim.setObjectPosition(target,-1,sim.getObjectPosition(tip,-1))
        tipPosRel=sim.getObjectPosition(tip,youBotRef)--youBot)
        desiredPos={tipPosRel[1]-initialTipPosRelative[1],tipPosRel[2]-initialTipPosRelative[2],tipPosRel[3]-initialTipPosRelative[3]}
        for i=1,3,1 do 
            currentPos[i]=desiredPos[i]
        end
        -- Close any IK warning dialogs:
        if (ikFailedReportHandle>=0) then
            sim.endDialog(ikFailedReportHandle) -- We close any report message about IK failure
            ikFailedReportHandle=-1
        end
    end
    
    -- Now update the user interface:
    -- First the FK part, text boxes:
    for i=1,5,1 do
        simSetUIButtonLabel(ui,212+i,string.format("%.1f",sim.getJointPosition(armJoints[i])*180/math.pi))
    end
    -- Then the FK part, sliders, based on the target joint position if in FK mode, or based on the current joint position if in IK mode:
    for i=1,5,1 do
        cyclic,interval=sim.getJointInterval(armJoints[i])
        if (ikMode) then
            simSetUISlider(ui,202+i,1000*(sim.getJointPosition(armJoints[i])-interval[1])/interval[2])
        else
            simSetUISlider(ui,202+i,1000*(desiredJ[i]-interval[1])/interval[2])
        end
    end
    
    -- Now the IK part:
    -- First the text boxes:
    for i=1,3,1 do
        str=string.format("%.3f",currentPos[i])
        if (str=='-0.000') then
            str='0.000' -- avoid having the - sign appearing and disappearing when 0
        end
        simSetUIButtonLabel(ui,217+i,str)
    end
    -- Now the sliders, based on the desired position if in IK mode, or based on the current tip position if in FK mode:
    for i=1,3,1 do
        if (ikMode) then
            simSetUISlider(ui,207+i,1000*(desiredPos[i]-ikMinPos[i])/ikRange[i])
        else
            simSetUISlider(ui,207+i,1000*(currentPos[i]-ikMinPos[i])/ikRange[i])
        end
    end
    
    -- Now apply the desired wheel velocities:
    sim.setJointTargetVelocity(wheelJoints[1],-forwBackVel-leftRightVel-rotVel)
    sim.setJointTargetVelocity(wheelJoints[2],-forwBackVel+leftRightVel-rotVel)
    sim.setJointTargetVelocity(wheelJoints[3],-forwBackVel-leftRightVel+rotVel)
    sim.setJointTargetVelocity(wheelJoints[4],-forwBackVel+leftRightVel+rotVel)
end 

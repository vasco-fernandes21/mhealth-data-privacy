from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..ml.engine import FederatedSimulation
import asyncio
from typing import Dict

router = APIRouter()

# Store active simulations (important for stopping them)
active_simulations: Dict[str, FederatedSimulation] = {}


@router.websocket("/ws/training/{client_id}")
async def training_endpoint(websocket: WebSocket, client_id: str):
    # Ensure this is called first to establish the connection
    await websocket.accept()
    print(f"WebSocket client connected: {client_id}")
    
    # Initialize sim to None initially
    sim = None

    try:
        # Send a connection confirmation log immediately
        await websocket.send_json({"type": "log", "message": "WebSocket connected. Ready for commands."})
        
        while True:
            # This line will block until a message is received
            try:
                data = await websocket.receive_json()
            except Exception as e:
                print(f"Error receiving JSON from {client_id}: {e}")
                break
            
            if data.get('action') == 'start':
                # Check if a simulation is already running for this client_id
                if client_id in active_simulations and active_simulations[client_id].should_stop == False:
                    await websocket.send_json({"type": "error", "message": "A simulation is already active for this client."})
                    continue  # Wait for next message
                
                print(f"Starting simulation for {client_id} with config: {data.get('config')}")
                try:
                    # Create and store new simulation
                    sim = FederatedSimulation(
                        dataset_name=data['config']['dataset'],
                        n_clients=data['config']['clients'],
                        sigma=data['config']['sigma']
                    )
                    active_simulations[client_id] = sim  # Store reference

                    # Callback to push updates to WS
                    async def update_ui(msg):
                        try:
                            await websocket.send_json(msg)
                        except RuntimeError as e:
                            # Handle cases where websocket might have closed mid-simulation
                            print(f"Warning: Could not send to client {client_id}: {e}")
                            if sim:
                                sim.stop()  # Attempt to stop the server-side sim
                            raise  # Re-raise to break the outer while loop

                    # Run simulation in a background task
                    # IMPORTANT: Use asyncio.create_task to run without blocking the WebSocket message receiver loop
                    simulation_task = asyncio.create_task(sim.run_simulation(update_ui))
                    
                    # Wait for either simulation to complete OR a stop message
                    # Use asyncio.wait to handle both concurrently
                    message_received = asyncio.Event()
                    received_data = {'action': None}
                    
                    async def wait_for_message():
                        try:
                            msg = await websocket.receive_json()
                            received_data['action'] = msg.get('action')
                            message_received.set()
                        except:
                            message_received.set()
                    
                    message_task = asyncio.create_task(wait_for_message())
                    
                    # Wait for either simulation or message
                    done, pending = await asyncio.wait(
                        [simulation_task, message_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel the pending task
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    # Check what completed
                    if simulation_task in done:
                        try:
                            await simulation_task  # Get result/exception
                        except Exception as e:
                            print(f"Simulation error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })
                        # Clean up
                        if client_id in active_simulations:
                            del active_simulations[client_id]
                        break
                    elif message_task in done:
                        # Process the received message
                        if received_data['action'] == 'stop':
                            sim.stop()
                            simulation_task.cancel()
                            try:
                                await simulation_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            await websocket.send_json({"type": "log", "message": "Simulation stopped."})
                            if client_id in active_simulations:
                                del active_simulations[client_id]
                            break
                        # If it's another action, continue loop
                        data = {'action': received_data['action']}
                        continue
                    
                except Exception as e:
                    print(f"Error during simulation setup for {client_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                
            elif data.get('action') == 'stop':
                if client_id in active_simulations:
                    active_simulations[client_id].stop()
                    await websocket.send_json({"type": "log", "message": "Simulation stop requested."})
                else:
                    await websocket.send_json({"type": "error", "message": "No active simulation found to stop."})
            
            # Additional actions could be handled here (e.g., 'pause', 'get_status')

    except WebSocketDisconnect:
        # Clean up on disconnect
        if sim and client_id in active_simulations and active_simulations[client_id] == sim:
            sim.stop()
            del active_simulations[client_id]
        print(f"Client {client_id} disconnected")
    except RuntimeError as e:
        # Catch errors from the callback if WS closed unexpectedly
        print(f"Runtime error in WebSocket for client {client_id}: {e}")
        if sim and client_id in active_simulations and active_simulations[client_id] == sim:
            sim.stop()
            if client_id in active_simulations:
                del active_simulations[client_id]
    except Exception as e:
        # Catch any other unexpected errors during message processing
        print(f"Unexpected error in WebSocket for client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        if sim and client_id in active_simulations and active_simulations[client_id] == sim:
            sim.stop()
            if client_id in active_simulations:
                del active_simulations[client_id]
        try:
            await websocket.close(code=1011)  # Internal error
        except:
            pass

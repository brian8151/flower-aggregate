# Copyright 2024 ONYX Aikya. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pydantic import BaseModel, Field
from typing import Any, Dict

class MessageRequest(BaseModel):
    message_id: str = Field(..., description="Unique identifier for the message")
    client_id: str = Field(..., description="Identifier for the client sending the message")
    strategy: str = Field(..., description="Federated learning strategy")
    parameters: Any  = Field(..., description="Parameters for the model weight")
    metrics: Dict[str, float] = Field(..., description="Metrics reporting from client")
    num_examples: int = Field(..., description="Number of examples used in the client's dataset")
    properties: Dict[str, Any] = Field(..., description="Additional properties")
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for NVIDIAVLMRerank and related VLM reranker helpers."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from nvidia_rag.utils.vlm_reranker import (
    NVIDIAVLMRerank,
    _build_vlm_rerank_invoke_url,
)


class TestVLMRerankInvokeUrl:
    """Tests for VLM reranker endpoint selection."""

    def test_build_invoke_url_for_hosted_endpoint(self):
        """Hosted NVIDIA endpoints should use the retrieval reranking path."""
        url = _build_vlm_rerank_invoke_url(
            "https://ai.api.nvidia.com/v1",
            "nvidia/llama-nemotron-rerank-vl-1b-v2",
        )

        assert (
            url
            == "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"
        )

    def test_build_invoke_url_for_self_hosted_endpoint(self):
        """Self-hosted NIMs should use the OpenAI-style ranking path."""
        url = _build_vlm_rerank_invoke_url(
            "nemotron-ranking-vl-ms:8000",
            "nvidia/llama-nemotron-rerank-vl-1b-v2",
        )

        assert url == "http://nemotron-ranking-vl-ms:8000/v1/ranking"


class TestNVIDIAVLMRerank:
    """Focused tests for the requests-based multimodal reranker."""

    def test_build_payload_attaches_image_when_available(self):
        """Payloads should include passage images when MinIO content is available."""
        ranker = NVIDIAVLMRerank(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000",
        )
        doc = Document(
            page_content="A chart about tax rates.",
            metadata={
                "content_metadata": {
                    "type": "image",
                    "page_number": 1,
                    "location": [0, 0, 1, 1],
                },
                "collection_name": "demo",
                "source": {
                    "source_id": "sample.pdf",
                    "source_location": "s3://default-bucket/demo/page.png",
                },
            },
        )

        with patch.object(ranker, "_build_image_data_url", return_value="data:image/png;base64,abc"):
            payload = ranker._build_payload("What does the chart show?", [doc])

        assert payload["query"] == {"text": "What does the chart show?"}
        assert payload["passages"] == [
            {
                "text": "A chart about tax rates.",
                "image": "data:image/png;base64,abc",
            }
        ]

    def test_compress_documents_returns_ranked_documents_in_api_order(self):
        """Returned documents should follow the API ranking order."""
        ranker = NVIDIAVLMRerank(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000",
            top_n=2,
        )
        docs = [
            Document(page_content="doc-0", metadata={}),
            Document(page_content="doc-1", metadata={}),
            Document(page_content="doc-2", metadata={}),
        ]
        mock_response = Mock()
        mock_response.json.return_value = {
            "rankings": [
                {"index": 2, "logit": 3.5},
                {"index": 0, "logit": 1.2},
            ]
        }
        mock_response.raise_for_status.return_value = None
        ranker._session = Mock()
        ranker._session.post.return_value = mock_response

        reranked = ranker.compress_documents(docs, "test query")

        assert [doc.page_content for doc in reranked] == ["doc-2", "doc-0"]
        assert reranked[0].metadata["relevance_score"] == 3.5
        assert reranked[1].metadata["relevance_score"] == 1.2

    def test_compress_documents_rejects_out_of_range_index(self):
        """Invalid API indices should raise a clear runtime error."""
        ranker = NVIDIAVLMRerank(
            model="nvidia/llama-nemotron-rerank-vl-1b-v2",
            url="http://nemotron-ranking-vl-ms:8000",
        )
        mock_response = Mock()
        mock_response.json.return_value = {"rankings": [{"index": 99, "logit": 1.0}]}
        mock_response.raise_for_status.return_value = None
        ranker._session = Mock()
        ranker._session.post.return_value = mock_response

        with pytest.raises(
            RuntimeError, match="invalid response from VLM reranker: index out of range"
        ):
            ranker.compress_documents([Document(page_content="doc", metadata={})], "query")

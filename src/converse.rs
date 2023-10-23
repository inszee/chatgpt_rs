use std::path::Path;

use tokio::{fs::File, io::AsyncWriteExt};
use tiktoken_rs::{get_chat_completion_max_tokens, ChatCompletionRequestMessage};

#[cfg(feature = "functions")]
use crate::functions::{
    CallableAsyncFunction, FunctionArgument, FunctionCall, FunctionValidationStrategy, GptFunction,
    GptFunctionHolder,
};
#[cfg(feature = "functions")]
use std::collections::HashMap;
#[cfg(feature = "functions")]
use thiserror::Error;
#[cfg(feature = "streams")]
use {crate::types::ResponseChunk, futures::Stream};

use crate::{
    client::ChatGPT,
    types::{ChatMessage, CompletionResponse, Role},
};

/// Stores a single conversation session, and automatically saves message history
pub struct Conversation {
    pub(crate) client: ChatGPT,
    /// All the messages sent and received, starting with the beginning system message
    pub history: Vec<ChatMessage>,
    /// Set to `true` if you want to automatically send all functions to API with each message.
    ///
    /// Functions are counted as tokens internally, so it is set to `false` by default.
    #[cfg(feature = "functions")]
    pub always_send_functions: bool,
    #[cfg(feature = "functions")]
    functions: HashMap<String, Box<dyn GptFunctionHolder>>,
    #[cfg(feature = "functions")]
    function_descriptors: Vec<serde_json::Value>,
}

impl Conversation {
    /// Constructs a new conversation from an API client and the introductory message
    pub fn new(client: ChatGPT, first_message: String) -> Self {
        Self {
            client,
            history: vec![ChatMessage {
                role: Role::System,
                content: first_message,
                #[cfg(feature = "functions")]
                function_call: None,
            }],
            #[cfg(feature = "functions")]
            functions: HashMap::with_capacity(4),
            #[cfg(feature = "functions")]
            always_send_functions: false,
            #[cfg(feature = "functions")]
            function_descriptors: Vec::with_capacity(4),
        }
    }

    /// Constructs a new conversation from a pre-initialized chat history
    pub fn new_with_history(client: ChatGPT, history: Vec<ChatMessage>) -> Self {
        Self {
            client,
            history,
            #[cfg(feature = "functions")]
            functions: HashMap::with_capacity(4),
            #[cfg(feature = "functions")]
            always_send_functions: false,
            #[cfg(feature = "functions")]
            function_descriptors: Vec::with_capacity(4),
        }
    }

    /// Rollbacks the history by 1 message, removing the last sent and received message.
    pub fn rollback(&mut self) -> Option<ChatMessage> {
        let last = self.history.pop();
        self.history.pop();
        last
    }

    /// Adds a function that can later be called by ChatGPT
    #[cfg(feature = "functions")]
    pub fn add_function<
        A: FunctionArgument + Send + Sync + 'static,
        C: CallableAsyncFunction<A> + Send + Sync + 'static,
    >(
        &mut self,
        prebuilt: GptFunction<A, C>,
    ) -> crate::Result<()> {
        self.function_descriptors
            .push(serde_json::to_value(&prebuilt.descriptor).map_err(crate::err::Error::from)?);
        self.functions
            .insert(prebuilt.descriptor.name.to_owned(), Box::new(prebuilt));
        Ok(())
    }

    /// Sends a message from a specified role to the ChatGPT API and returns the completion response.
    #[cfg_attr(feature = "functions", async_recursion::async_recursion)]
    pub async fn send_role_message<S: Into<String> + Send + Sync>(
        &mut self,
        role: Role,
        message: S,
    ) -> crate::Result<CompletionResponse> {
        self.history.push(ChatMessage {
            role,
            content: message.into(),
            #[cfg(feature = "functions")]
            function_call: None,
        });

        #[cfg(feature = "functions")]
        let resp = if self.always_send_functions {
            self.client
                .send_history_functions(&self.history, &self.function_descriptors)
                .await?
        } else {
            self.client.send_history(&self.history).await?
        };
        #[cfg(not(feature = "functions"))]
        let resp = self.client.send_history(&self.history).await?;
        let msg = &resp.message_choices[0].message;
        self.history.push(msg.clone());
        if let Some(function_response) = self.process_possible_function_response(msg).await {
            self.history
                .push(function_response.message_choices[0].message.clone());
            Ok(function_response)
        } else {
            Ok(resp)
        }
    }

    /// Sends the message to the ChatGPT API and returns the completion response.
    ///
    /// Execution speed depends on API response times.
    pub async fn send_message<S: Into<String> + Send + Sync>(
        &mut self,
        message: S,
    ) -> crate::Result<CompletionResponse> {
        self.send_role_message(Role::User, message).await
    }

    /// Sends a message with all functions to the ChatGPT API and returns the completion response.
    ///
    /// **NOTE**: Functions are counted as tokens internally.
    #[cfg(feature = "functions")]
    pub async fn send_message_functions<S: Into<String>>(
        &mut self,
        message: S,
    ) -> crate::Result<CompletionResponse> {
        self.history.push(ChatMessage {
            role: Role::User,
            content: message.into(),
            #[cfg(feature = "functions")]
            function_call: None,
        });
        let resp = self
            .client
            .send_history_functions(&self.history, &self.function_descriptors)
            .await?;
        let msg = &resp.message_choices[0].message;
        self.history.push(msg.clone());
        if let Some(function_response) = self.process_possible_function_response(msg).await {
            self.history
                .push(function_response.message_choices[0].message.clone());
            Ok(function_response)
        } else {
            Ok(resp)
        }
    }

    /// calculate current request message ticktoken
    fn get_retain_completion_max_tokens(&mut self) -> anyhow::Result<usize> {
        let model = self.client.config.engine.to_string();
        let messages: Vec<ChatCompletionRequestMessage> = self
            .history
            .iter()
            .map(|item| {
                let role_name = match item.role {
                    Role::System => "system".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::User => "user".to_string(),
                    Role::Function => "function".to_string()
                };
                ChatCompletionRequestMessage {
                    content: Some(item.content.clone()),
                    role: role_name,
                    name: None,
                    function_call: None,
                }
            })
            .collect();
        get_chat_completion_max_tokens(&model, &messages)
    }

    /// Sends a message with specified role to the ChatGPT API and returns the completion response as stream.
    ///
    /// Note, that this method will not automatically save the received message to history, as
    /// it is returned in streamed chunks. You will have to collect them into chat message yourself.
    ///
    /// You can use [`ChatMessage::from_response_chunks`] for this
    ///
    /// Requires the `streams` crate feature.
    #[cfg(feature = "streams")]
    pub async fn send_role_message_streaming<S: Into<String>>(
        &mut self,
        role: Role,
        message: S,
    ) -> crate::Result<impl Stream<Item = ResponseChunk>> {
        self.history.push(ChatMessage {
            role,
            content: message.into(),
            #[cfg(feature = "functions")]
            function_call: None,
        });

        if let Ok(token_count) = self.get_retain_completion_max_tokens() {
            log::info!("send_message_streaming request tiktoken = {}", token_count);
            if token_count <= 1024 {
                // TODO: save user chat history into files.
                // 给completion max token保留最少1024的长度
                let mut new_history = vec![];
                // 保留3轮对话
                if self.history.len() <= 3 {
                    log::warn!("send_message_streaming too long for user input...");
                } else {
                    // 只保留前3轮对话主题和当前对话
                    log::warn!("send_message_streaming transcate token for input...");
                    new_history.push(self.history[0].to_owned());
                    new_history.push(self.history[1].to_owned());
                    new_history.push(self.history[2].to_owned());
                    if new_history.len() == 4 {
                        new_history.push(self.history[self.history.len() - 1].to_owned());
                    } else if new_history.len() == 5 {
                        new_history.push(self.history[self.history.len() - 2].to_owned());
                        new_history.push(self.history[self.history.len() - 1].to_owned());
                    }  else if new_history.len() == 6 {
                        new_history.push(self.history[self.history.len() - 3].to_owned());
                        new_history.push(self.history[self.history.len() - 2].to_owned());
                        new_history.push(self.history[self.history.len() - 1].to_owned());
                    } else {
                        // 保留多一轮对话(user + assistant)
                        new_history.push(self.history[self.history.len() - 4].to_owned());
                        new_history.push(self.history[self.history.len() - 3].to_owned());
                        new_history.push(self.history[self.history.len() - 2].to_owned());
                        new_history.push(self.history[self.history.len() - 1].to_owned());
                    }
                    self.history.clear();
                    self.history.extend_from_slice(&new_history);
                }
            }
        }
        let stream = self.client.send_history_streaming(&self.history).await?;
        Ok(stream)
    }

    /// Sends the message to the ChatGPT API and returns the completion response as stream.
    ///
    /// Note, that this method will not automatically save the received message to history, as
    /// it is returned in streamed chunks. You will have to collect them into chat message yourself.
    ///
    /// You can use [`ChatMessage::from_response_chunks`] for this
    ///
    /// Requires the `streams` crate feature.
    #[cfg(feature = "streams")]
    pub async fn send_message_streaming<S: Into<String>>(
        &mut self,
        message: S,
    ) -> crate::Result<impl Stream<Item = ResponseChunk>> {
        self.send_role_message_streaming(Role::User, message).await
    }

    /// Saves the history to a local JSON file, that can be restored to a conversation at runtime later.
    #[cfg(feature = "json")]
    pub async fn save_history_json<P: AsRef<Path>>(&self, to: P) -> crate::Result<()> {
        let path = to.as_ref();
        if path.exists() {
            tokio::fs::remove_file(path).await?;
        }
        let mut file = File::create(path).await?;
        file.write_all(&serde_json::to_vec(&self.history)?).await?;
        Ok(())
    }

     /// if request error and u can push history after error response.
     pub fn push_history_after_streaming<S: Into<String>>(
        &mut self,
        role: Role,
        message: S,
    ) -> crate::Result<()> {
        self.history.push(ChatMessage {
            role: role,
            content: message.into(),
            function_call: None,
        });
        Ok(())
    }

    /// Saves the history to a local postcard file, that can be restored to a conversation at runtime later.
    #[cfg(feature = "postcard")]
    pub async fn save_history_postcard<P: AsRef<Path>>(&self, to: P) -> crate::Result<()> {
        let path = to.as_ref();
        if path.exists() {
            tokio::fs::remove_file(path).await?;
        }
        let mut file = File::create(path).await?;
        file.write_all(&postcard::to_allocvec(&self.history)?)
            .await?;
        Ok(())
    }

    #[cfg(not(feature = "functions"))]
    async fn process_possible_function_response(
        &mut self,
        _message: &ChatMessage,
    ) -> Option<CompletionResponse> {
        None
    }

    #[cfg(feature = "functions")]
    async fn process_possible_function_response(
        &mut self,
        message: &ChatMessage,
    ) -> Option<CompletionResponse> {
        if let Some(call) = &message.function_call {
            if let Some(Ok(result)) = self.process_function(call).await {
                Some(result)
            } else {
                None
            }
        } else {
            None
        }
    }

    // TODO: streamed function processing is technically possible
    #[cfg(feature = "functions")]
    async fn process_function(
        &mut self,
        call: &FunctionCall,
    ) -> Option<crate::Result<CompletionResponse>> {
        let call_result = if let Some(fnc) = self.functions.get(&call.name) {
            // TODO: better error handling?
            // TODO: maybe replace check for SerdeJsonError with a special error?
            fnc.try_invoke(&call.arguments).await.map_err(|err| {
                if let crate::err::Error::SerdeJsonError(_) = err {
                    FunctionCallError::InvalidArguments
                } else {
                    FunctionCallError::InnerError(err.to_string())
                }
            })
        } else {
            Err(FunctionCallError::InvalidFunction)
        };
        if let Ok(result) = call_result {
            let result = serde_json::to_string(&result);
            return Some(self.send_role_message(Role::Function, result.ok()?).await);
        }

        if self.client.config.function_validation == FunctionValidationStrategy::Strict {
            // Sending error response from function
            Some(
                self.send_role_message(Role::System, call_result.unwrap_err().to_string())
                    .await,
            )
        } else {
            None
        }
    }
}

#[cfg(feature = "functions")]
#[derive(Debug, Clone, Error)]
enum FunctionCallError {
    #[error("Invalid function call: invalid arguments given to this function")]
    InvalidArguments,
    #[error("Invalid function call: this function does not exist")]
    InvalidFunction,
    #[error("Exception encountered when calling function: {0}")]
    InnerError(String),
}

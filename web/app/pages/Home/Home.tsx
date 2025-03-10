import { Box, Button, Container, MenuItem, Stack, TextField, Typography } from "@mui/material";
import { Header } from "../../components/Header";
import { CaptchaDisplay } from "~/components/CaptchaDisplay";
import { CaptchaGallery } from "~/components/CaptchaGallery";
import { GenerateCaptcha, GetHistory, SolveCaptcha, type GetHistoryResponse } from "~/lib/api";
import { MODEL_OPTIONS } from "~/lib/constant";
import { useEffect, useMemo, useState } from "react";


export function Home() {
  const [generateStatus, setGenerateStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [captchaImageBlob, setCaptchaImageBlob] = useState<Blob | null>(null);
  const [captchaImageUrl, setCaptchaImageUrl] = useState<string | null>(null);
  const [captchaText, setCaptchaText] = useState<string>("");
  const [generateCaptchaError, setGenerateCaptchaError] = useState<string | null>(null);

  // Generate Form
  const [captionInput, setCaptionInput] = useState<string>("");
  const [captionInputError, setCaptionInputError] = useState<boolean>(false);

  const handleGenerateCaptcha = (text: string) => {
    GenerateCaptcha({
      text,
      onLoading: () => {
        console.log("Loading...");
        setGenerateStatus("loading");
      },
      onSuccess: (response) => {
        console.log(response.data);
        const captchaImageUrl = URL.createObjectURL(response.data);
        setCaptchaText(text);
        setCaptchaImageBlob(response.data);
        setCaptchaImageUrl(captchaImageUrl);
        setGenerateStatus("success");
      },
      onError: (error) => {
        console.error(error);
        setCaptchaImageUrl(null);
        setGenerateCaptchaError(error.message);
        setGenerateStatus("error");
      },
    });
  };

  const [solvedCaptchaText, setSolvedCaptchaText] = useState<string | null>(null);
  const [solvedCaptchaModel, setSolvedCaptchaModel] = useState<string | null>(null);

  // Solve Form
  const [model, setModel] = useState<string>(MODEL_OPTIONS[2].value);

  const handleSolveCaptcha = () => {
    if (!captchaImageBlob) {
      console.error("Captcha image blob is not available.");
      return;
    } else {
      SolveCaptcha({
        imageBlob: captchaImageBlob,
        model,
        label: captchaText,
        onLoading: () => {
          console.log("Loading...");
        },
        onSuccess: (response) => {
          console.log(response.data);
          setSolvedCaptchaModel(model);
          setSolvedCaptchaText(response.data.decoded);
          fetchHistory();
        },
        onError: (error) => {
          console.error(error);
          setSolvedCaptchaModel(null);
          setSolvedCaptchaText(null);
        },
      });
    }
  };

  // Fetch History
  const [history, setHistory] = useState<GetHistoryResponse[]>([]);
  const fetchHistory = () => {
    // Fetch history from the API
    GetHistory({
      onLoading: () => {
        console.log("Loading...");
      },
      onSuccess: (response) => {
        console.log(response.data);
        setHistory(response.data.results);
      },
      onError: (error) => {
        console.error(error);
      },
    });
  };
  useEffect(() => {
    fetchHistory();
  }, [])

  return (
    <>
      {/* Header */}
      <Header />
      {/* Content */}
      <Container
        maxWidth="lg"
        sx={{
          py: 5,
        }}
      >
        <Stack
          spacing={4}
        >
          <Typography>
            This is a demo page for solving randomly generated captchas using machine learning models. <br />
            The 3 models available are <strong>CNN (Convolution Neural Network)</strong>, <strong>CRNN (Convolutional Recurrent Neural Network)</strong>, and <strong>ViT (Vision Transformer)</strong>.
          </Typography>
          {/* Section 1 */}
          <Box>
            {/* Section 1 Title */}
            <Typography
              variant="h4"
              mb={1}
            >
              Step1: Generate Captcha
            </Typography>
            <Typography
              mb={2}
            >
              Enter a 5-digit alphanumeric text to randomly generate a captcha image.
            </Typography>
            {/* Generate Captcha Form */}
            <Box
              sx={{
                display: "flex",
                flexDirection: "row",
                gap: 2,
              }}
            >
              {/* Generate Form */}
              <Box
                component="form"
                width={400}
                onSubmit={(e) => {
                  e.preventDefault();
                  const text = (e.target as any)["captcha-text"].value;
                  handleGenerateCaptcha(text);
                }}
              >
                <Stack
                  spacing={1}
                >
                  <TextField
                    required
                    id="captcha-text"
                    label="Captcha Text"
                    variant="outlined"
                    placeholder="Uppercase 5-digit alphanumeric text"
                    fullWidth
                    value={captionInput}
                    onChange={(e) => {
                      setCaptionInputError(false);
                      setCaptionInput(e.target.value);
                    }}
                    onInvalid={() => setCaptionInputError(true)}
                    error={captionInputError}
                    slotProps={{
                      htmlInput: {
                        minLength: 5,
                        maxLength: 5,
                        style: {
                          textTransform: "uppercase",
                        },
                        pattern: "[A-Za-z0-9]{5}",
                      }
                    }}
                  />
                  <Button
                    variant="contained"
                    color="primary"
                    type="submit"
                    fullWidth
                  >
                    Generate
                  </Button>
                </Stack>
              </Box>
              {/* Captcha display */}
              <Box
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                }}
              >
                {generateStatus === "error" ? (
                  <Typography
                    color="error"
                  >
                    Error: {generateCaptchaError}
                  </Typography>
                ) : (
                  <Typography>
                    Caption: <strong>{captchaText.toUpperCase()}</strong>
                  </Typography>
                )}
                <CaptchaDisplay
                  captchaText={captchaText}
                  captchaImageUrl={captchaImageUrl ? captchaImageUrl : undefined}
                  status={generateStatus}
                />
              </Box>
            </Box>
          </Box>
          {/* Section 2 */}
          <Box>
            {/* Section 2 Title */}
            <Typography
              variant="h4"
              mb={1}
            >
              Step2: Solve Captcha
            </Typography>
            <Typography
              mb={1}
            >
              Choose the model use to solve the generated captcha.<br />
              The default model is <strong>(Vision Transformer)</strong>.<br />
            </Typography>
            <Typography
              mb={2}
              variant="body2"
              color="text.secondary"
            >
              You need to first generate a captcha before solving it.
            </Typography>
            <Box
              sx={{
                display: "flex",
                flexDirection: "row",
                gap: 2,
              }}
            >
              {/* Generate Captcha Form */}
              <Box
                component="form"
                width={400}
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSolveCaptcha();
                }}
              >
                <Stack
                  spacing={1}
                >
                  <TextField
                    id="model-selection"
                    label="Model Selection"
                    select
                    fullWidth
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  >
                    {MODEL_OPTIONS.map((option) => (
                      <MenuItem
                        key={option.value}
                        value={option.value}
                      >
                        {option.label}
                      </MenuItem>
                    ))}
                  </TextField>
                  <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    type="submit"
                    disabled={generateStatus !== "success"}
                  >
                    Solve
                  </Button>
                </Stack>
              </Box>
              {/* Display Solved Text */}
              <Box>
                <Typography>
                  Model Used: <strong>{solvedCaptchaModel}</strong><br />
                  Solved Text: <strong>{solvedCaptchaText ? `"${solvedCaptchaText}"` : "Not Available"}</strong>
                </Typography>
              </Box>
            </Box>
          </Box>
          {/* Section 3 */}
          <Box>
            <Typography
              variant="h4"
              mb={1}
            >
              Solved Captcha Gallery
            </Typography>
            <Typography
              mb={2}
            >
              Explore recently solved captchas (up to 30).
            </Typography>
            <CaptchaGallery
              items={
                history.map((item) => {
                  return {
                    captchaText: item.label,
                    captchaSolvedText: item.prediction,
                    captchaImageUrl: item.image,
                    model: item.model,
                  }
                })
              }
            />
          </Box>
          {/* Footer */}
          <Typography
            variant="body2"
            color="text.secondary"
            textAlign="center"
          >
            © 2025 Captcha Solver Demo by Yu-Kai "Steven" Wang (Isaac the Man)
          </Typography>
        </Stack>
      </Container>
    </>
  );
}
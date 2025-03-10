import { Box, Skeleton, Typography } from "@mui/material";


export interface CaptchaDisplayProps {
  captchaText: string;
  captchaImageUrl?: string;
  status: "idle" | "loading" | "success" | "error";
}

export function CaptchaDisplay({
  captchaText,
  captchaImageUrl,
  status = "idle",
}: CaptchaDisplayProps) {
  return (
    <>
      {status === "idle" || status === "error" ? (
        <Box
          component="div"
          width={160}
          height={60}
          bgcolor={"black"}
          color={"white"}
          position={"relative"}
        >
          <Typography
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
            }}
            textAlign={"center"}
          >
            {status === "idle" ? "Captcha Placholder" : "Render Error"}
          </Typography>
        </Box>
      ) : status === "loading" ? (
        <Skeleton variant="rectangular" width={160} height={60} />
      ) : status === "success" ? (
        <Box
          component="img"
          src={captchaImageUrl}
          alt={`Captcha Image: ${captchaText}`}
          width={160}
          height={60}
        />
      ) : null}
    </>
  );
}